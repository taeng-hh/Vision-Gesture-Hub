"""Optional asynchronous RGB assistance; base game and environment remain usable."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys,time
import numpy as np
from ksl.predict import prediction_event


def build_rois(samples, limit=16):
    """Same unmirrored hand/body crop geometry as the isolated RGB extractor."""
    import cv2
    if len(samples)<4:raise ValueError('Need at least four aligned video frames')
    chosen=np.unique(np.rint(np.linspace(0,len(samples)-1,min(limit,len(samples)))).astype(int))
    images=[];quality=[];motion=[];last_boxes=[None,None]
    def crop(image,box):
        cx,cy,side=box;side=max(2,int(np.ceil(side)))
        x0,y0=int(round(cx-side/2)),int(round(cy-side/2));h,w=image.shape[:2]
        l,t,r,b=max(0,x0),max(0,y0),min(w,x0+side),min(h,y0+side)
        canvas=np.full((side,side,3),114,np.uint8)
        if l<r and t<b:canvas[t-y0:b-y0,l-x0:r-x0]=image[t:b,l:r]
        return cv2.cvtColor(cv2.resize(canvas,(224,224),interpolation=cv2.INTER_AREA if side>224 else cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
    for index in chosen:
        _,image,f=samples[int(index)];f=np.asarray(f)
        if f.shape!=(67,3) or not np.isfinite(f).all() or min(f[44,2],f[47,2])<.2:raise ValueError('Invalid shoulders')
        center=(f[44,:2]+f[47,:2])/2;scale=float(np.linalg.norm(f[44,:2]-f[47,:2]))
        if scale<20:raise ValueError('Shoulder width too small')
        crops=[];q=[];m=[]
        for slot in range(2):
            hand=f[slot*21:slot*21+21];valid=hand[:,2]>=.2;wrist=f[49 if slot==0 else 46,:2]
            if valid.sum()>=15 and valid[0]:
                pts=hand[valid,:2];lo,hi=pts.min(0),pts.max(0);c=(lo+hi)/2;side=max(80.,1.55*float(np.max(hi-lo)))
                if last_boxes[slot] is not None:
                    c=.75*c+.25*np.array(last_boxes[slot][:2]);side=max(side,2.3*float(np.max(np.abs(pts-c))))
                box=(float(c[0]),float(c[1]),side);wrist=hand[0,:2];confidence=1.
            else:
                # Missing direct hand ROI stays missing: do not invent fingers.
                box=None;confidence=0.
            if box:
                last_boxes[slot]=box;crops.append(crop(image,box));pos=np.clip((wrist-center)/scale,-5,5)
                m.extend([float(pos[0]),float(pos[1]),box[2]/scale,confidence])
            else:crops.append(np.full((224,224,3),114,np.uint8));m.extend([0.,0.,0.,0.])
            q.append(confidence)
        crops.append(crop(image,(float(center[0]),float(center[1]+.25*scale),2.4*scale)))
        images.append(crops);quality.append(q);motion.append(m)
    if not np.any(quality):raise ValueError('No direct hand ROI')
    return dict(images=np.asarray(images,dtype=np.uint8),quality=np.asarray(quality,np.float32),
                motion=np.asarray(motion,np.float32),sample_positions=np.rint(np.linspace(0,len(images)-1,64)).astype(np.int64))


def merge_assistance(pair,rgb,classes):
    """Only same-candidate, confident RGB evidence may rescue a rejected command."""
    base=pair['game'];scores=np.asarray(pair['probabilities'],dtype=np.float64)
    rgb=np.asarray(rgb,dtype=np.float64)
    if rgb.shape!=scores.shape or not np.isfinite(rgb).all() or (rgb<0).any() or not np.isclose(rgb.sum(),1,atol=1e-4):
        raise ValueError('Invalid RGB probabilities')
    order=np.argsort(-rgb);best=int(order[0]);margin=float(rgb[best]-rgb[int(order[1])])
    if classes[best]['class_id']!=base['candidate_id']:return pair,'CNN 후보 다름 · 키포인트 유지'
    if rgb[best]<.75 or margin<.15:return pair,'CNN 점수 부족 · 키포인트 유지'
    mixed=.8*scores+.2*rgb
    event=prediction_event(classes,mixed,game_only=True)
    if not event['accepted'] or event['candidate_id']!=base['candidate_id']:return pair,'CNN 확인 후에도 보류'
    result=dict(pair);result['game']=event;result['assistance_used']=True
    # Keep the unmodified normal branch for comparison and fast-path gating.
    return result,'CNN 동일 후보 확인'


class GameCnnAssist:
    max_age=1.5
    def __init__(self, folder, classes, enabled=False):
        self.folder=Path(folder);self.classes=classes;self.enabled=enabled
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='ksl-cnn')
        self.pending=None;self.generation=0;self.branch=None;self.error=None
        self.status='CNN 보조 대기' if enabled else '키포인트 단독'
        self.next_request=0.

    def invalidate(self):
        self.generation+=1
        if self.pending is not None:self.pending['future'].cancel()
        self.pending=None

    def set_enabled(self, enabled):
        self.invalidate();self.enabled=enabled;self.error=None
        self.status='CNN 보조 대기' if enabled else '키포인트 단독'

    def _infer(self,samples):
        if self.branch is None:
            # Import only the RGB branch; existing core/ksl modules keep precedence.
            if not (self.folder/'fusion/predict.py').is_file():raise RuntimeError('CNN 코드 폴더 없음')
            if str(self.folder) not in sys.path:sys.path.append(str(self.folder))
            from fusion.predict import RGBBranch,align_probabilities
            self.branch=RGBBranch(self.folder/'models/rgb_ksl30')
            self.align=align_probabilities
        probabilities=self.branch.predict_proba(build_rois(samples))
        return self.align(probabilities,self.branch.ids,[c['class_id'] for c in self.classes])

    def evaluate(self,pair,samples,now,window_seconds):
        if not self.enabled:return pair
        if pair['normal']['accepted'] and not next((c.get('game_action') for c in self.classes if c['class_id']==pair['normal']['candidate_id']),None):
            if self.pending is not None:self.invalidate()
            self.status='비게임 수어 확정 · CNN 생략'
            return pair
        if pair['game']['accepted']:
            # Confident base commands never wait on the worker.
            if self.pending is not None:self.invalidate()
            self.status='키포인트 확정 · CNN 생략'
            return pair
        result=pair
        if self.pending is not None and self.pending['future'].done():
            job=self.pending;self.pending=None
            try:
                rgb=job['future'].result()
                if job['generation']==self.generation and now-job['time']<=self.max_age and pair['game']['candidate_id']==job['candidate']:
                    # Use the SAME historical keypoint/RGB clip, never mix old RGB
                    # with new keypoint probabilities. Current candidate must agree.
                    result,self.status=merge_assistance(job['pair'],rgb,self.classes)
                else:self.status='지난 CNN 결과 폐기'
            except ValueError:
                self.status='CNN 영상 추적 부족 · 키포인트 유지'
                self.next_request=now+.5
                return pair
            except Exception as exc:
                self.error=f'{type(exc).__name__}: {exc}';self.enabled=False
                self.status='CNN 사용 불가 · 키포인트로 복귀 (CNN용 실행 파일 확인)'
                return pair
        if self.pending is None and now>=self.next_request and self.enabled:
            chosen=[s for s in samples if now-window_seconds-1e-6<=s[0]<=now]
            if len(chosen)>=4:
                # Bounded copies are owned by the worker and never saved to disk.
                indices=np.unique(np.rint(np.linspace(0,len(chosen)-1,min(16,len(chosen)))).astype(int))
                snapshot=[(chosen[i][0],chosen[i][1].copy(),chosen[i][2].copy()) for i in indices]
                self.pending=dict(future=self.executor.submit(self._infer,snapshot),time=now,generation=self.generation,
                                  candidate=pair['game']['candidate_id'],pair=pair)
                self.next_request=now+.5
                if result is pair:self.status='CNN 확인 중 · 게임은 계속 진행'
            elif result is pair:self.status='CNN 영상 입력 대기'
        return result

    def close(self):
        self.invalidate();self.executor.shutdown(wait=False,cancel_futures=True)
