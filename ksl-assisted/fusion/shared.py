"""Original-resolution hand crops, upper-body context and wrist motion for RGB KSL."""
from contextlib import ExitStack
from itertools import permutations
from pathlib import Path
import math
from types import SimpleNamespace
from core.live_sign import webcam_frame
import cv2
import numpy as np
import torch

STEPS = 64
DIM = 1736
SPEC = 'rgb-handroi3-mobilenetv3small-motion8-v1-64x1736'


def square_crop(frame, box):
    cx, cy, side = box
    side = max(2, int(math.ceil(side)))
    x0, y0 = int(round(cx-side/2)), int(round(cy-side/2))
    x1, y1 = x0+side, y0+side
    h, w = frame.shape[:2]
    l,t,r,b = max(0,x0),max(0,y0),min(w,x1),min(h,y1)
    canvas = np.full((side,side,3),114,np.uint8)
    if l<r and t<b:
        canvas[t-y0:b-y0,l-x0:r-x0] = frame[t:b,l:r]
    resized = cv2.resize(canvas,(224,224),interpolation=cv2.INTER_AREA if side>224 else cv2.INTER_LINEAR)
    return cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)


def preview(frame, crops, boxes, path):
    h,w = frame.shape[:2]
    image = cv2.resize(frame,(640,360))
    for name, box, color in zip(['L','R','Body'],boxes,[(40,210,40),(30,30,230),(230,140,30)]):
        if box is None: continue
        cx,cy,s = box
        a=(int((cx-s/2)*640/w),int((cy-s/2)*360/h))
        b=(int((cx+s/2)*640/w),int((cy+s/2)*360/h))
        cv2.rectangle(image,a,b,color,2)
        cv2.putText(image,name,(max(0,a[0]),max(14,a[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    canvas=np.full((650,704,3),245,np.uint8)
    cv2.putText(canvas,'Original frame -> original-resolution ROI crops',(10,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
    canvas[30:390,32:672] = image
    for j,(name,crop) in enumerate(zip(['Left hand','Right hand','Upper body'],crops)):
        x=j*240
        cv2.putText(canvas,name,(x+5,411),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
        canvas[420:644,x:x+224] = cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path),canvas)


class SharedExtractor:
    """One shared pixel preprocessing pass, independent of CNN weights."""
    def __init__(self,assets):
        self.assets=Path(assets)

    def extract(self,path,start=0.,end=None,preview_path=None,roi_cache_path=None):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        cap=path.capture() if isinstance(path,FrameClip) else cv2.VideoCapture(str(path))
        if not cap.isOpened():raise ValueError('Cannot open '+str(path))
        fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps<=0 or n<=0:cap.release();raise ValueError('Invalid video duration')
        end=n/fps if end is None else end
        if not 0<=start<end<=n/fps+1/fps:cap.release();raise ValueError('Invalid interval')
        first=max(0,math.ceil(start*fps-1e-6));last=min(n-1,math.ceil(end*fps-1e-6)-1)
        if first>last:cap.release();raise ValueError('Empty interval')
        indices=np.rint(np.linspace(first,last,STEPS)).astype(np.int64)
        wanted=set(indices.tolist());entries={};last_boxes=[None,None];last_pose=None
        unique_qualities=[];box_sizes=[];sample_debug=[]
        try:
            with ExitStack() as stack:
                hands=stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(self.assets/'hand_landmarker.task')),
                    running_mode=vision.RunningMode.VIDEO,num_hands=2)))
                pose_task=stack.enter_context(vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(self.assets/'pose_landmarker_lite.task')),
                    running_mode=vision.RunningMode.VIDEO,num_poses=1)))
                for index in range(last+1):
                    ok,frame=cap.read()
                    if not ok:raise ValueError('Video truncated')
                    if index not in wanted:continue
                    h,w=frame.shape[:2]
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                    timestamp=round(index*1000/fps)
                    detected=hands.detect_for_video(image,timestamp)
                    pr=pose_task.detect_for_video(image,timestamp)
                    confidence=lambda p:min(float(p.visibility or 0),float(p.presence or 0))
                    pp=pr.pose_landmarks[0] if pr.pose_landmarks else None
                    if pp and min(confidence(pp[11]),confidence(pp[12]))>=.5:
                        xy=np.array([[p.x*w,p.y*h] for p in pp],np.float32)
                        center=(xy[11]+xy[12])/2;scale=float(np.linalg.norm(xy[11]-xy[12]))
                        if scale>=20:last_pose=(xy,center,scale,index)
                    if last_pose is None or (index-last_pose[3])/fps>.5:
                        raise ValueError('No reliable upper-body reference')
                    xy,center,scale,_=last_pose
                    body_box=(float(center[0]),float(center[1]+.25*scale),2.4*scale)
                    hand_points=[np.array([[p.x*w,p.y*h] for p in hs],np.float32) for hs in detected.hand_landmarks]
                    slots={}
                    if hand_points:
                        costs=[]
                        for assignment in permutations(range(2),len(hand_points)):
                            distances=[float(np.linalg.norm(points[0]-xy[(15,16)[slot]])/scale)
                                       for points,slot in zip(hand_points,assignment)]
                            costs.append((sum(distances),assignment,distances))
                        costs.sort()
                        for k,slot in enumerate(costs[0][1]):
                            if costs[0][2][k]<=.9:slots[slot]=hand_points[k]
                    crops=[];boxes=[];motion=[];qualities=[];debug=[]
                    for slot in range(2):
                        points=slots.get(slot);wrist=xy[(15,16)[slot]]
                        if points is not None:
                            lo,hi=points.min(0),points.max(0);c=(lo+hi)/2
                            side=max(80.,1.55*float(np.max(hi-lo)))
                            if last_boxes[slot] is not None:
                                c=.75*c+.25*np.array(last_boxes[slot][:2])
                                side=max(side,2.3*float(np.max(np.abs(points-c))))
                            box=(float(c[0]),float(c[1]),side)
                            wrist=points[0];quality=1.;kind='hand_landmarks'
                            span=(hi-lo).tolist()
                        elif pp and confidence(pp[(15,16)[slot]])>=.3:
                            elbow=xy[(13,14)[slot]];c=wrist+.35*(wrist-elbow)
                            side=max(100.,.65*scale,1.4*float(np.linalg.norm(wrist-elbow)))
                            box=(float(c[0]),float(c[1]),side);quality=.5;kind='pose_wrist_fallback';span=None
                        else:
                            box=None;quality=0.;kind='missing';span=None
                        if box is not None:
                            crop=square_crop(frame,box);last_boxes[slot]=box;box_sizes.append(box[2])
                            position=np.clip((wrist-center)/scale,-5,5)
                            motion.extend([float(position[0]),float(position[1]),float(box[2]/scale),quality])
                        else:
                            crop=np.full((224,224,3),114,np.uint8);motion.extend([0.,0.,0.,0.])
                        crops.append(crop);boxes.append(box);qualities.append(quality)
                        debug.append({'source':kind,'original_hand_span_xy':span,'original_crop_side':box[2] if box else None,
                                      'hand_span_after_old_full_frame_resize':[(v*224/max(w,h)) for v in span] if span else None})
                    crops.append(square_crop(frame,body_box));boxes.append(body_box)
                    observation=SimpleNamespace(hands=[SimpleNamespace(landmarks=hs) for hs in detected.hand_landmarks])
                    keypoints=webcam_frame(observation,pp,w,h)
                    if keypoints is None:
                        # Preserve a reliable body reference but do not fabricate finger joints.
                        keypoints=np.zeros((67,3),np.float32)
                        for mi,oi in ((11,5),(12,2),(15,7),(16,4)):
                            keypoints[42+oi]=[float(xy[mi,0]),float(xy[mi,1]),.5]
                    entries[index]=(crops,np.array(motion,np.float32),qualities,keypoints)
                    unique_qualities.append(qualities);sample_debug.append({'frame':index,'hands':debug})
                    if preview_path is not None and index==int(indices[len(indices)//2]):
                        preview(frame,crops,boxes,preview_path)
        finally:cap.release()
        unique=sorted(entries)
        all_crops=np.stack([crop for i in unique for crop in entries[i][0]])
        q=np.asarray(unique_qualities)
        if not np.any(q>0):raise ValueError('No hand ROI available')
        quality={'unique_source_frames':len(unique),'left_direct_detection_fraction':float(np.mean(q[:,0]==1)),
                 'right_direct_detection_fraction':float(np.mean(q[:,1]==1)),
                 'any_hand_roi_fraction':float(np.mean(np.any(q>0,axis=1))),
                 'left_pose_fallback_fraction':float(np.mean(q[:,0]==.5)),
                 'right_pose_fallback_fraction':float(np.mean(q[:,1]==.5)),
                 'mean_original_hand_crop_side':float(np.mean(box_sizes)),
                 'sample_frames':sample_debug}
        positions={i:j for j,i in enumerate(unique)}
        rois=dict(images=all_crops.reshape(len(unique),3,224,224,3),
                  motion=np.stack([entries[i][1] for i in unique]),
                  quality=np.array([entries[i][2] for i in unique],np.float32),
                  sample_positions=np.array([positions[int(i)] for i in indices],np.int64),
                  source_frame_indices=np.array(unique,np.int64),
                  keypoints=np.stack([entries[i][3] for i in unique]))
        if roi_cache_path is not None:
            import os
            destination=Path(roi_cache_path);destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:np.savez_compressed(stream,**rois)
            os.replace(temporary,destination)
        return rois,quality


class ROIExtractor:
    """Compatibility wrapper for the existing MobileNet hand-ROI model."""
    def __init__(self,cnn_weights,assets):
        from torchvision.models import mobilenet_v3_small
        torch.set_num_threads(2)
        self.cnn=mobilenet_v3_small(weights=None)
        self.cnn.load_state_dict(torch.load(cnn_weights,map_location='cpu',weights_only=True))
        self.cnn.eval();self.cnn.requires_grad_(False)
        self.cropper=SharedExtractor(assets)
        self.mean=torch.tensor([.485,.456,.406]).view(1,3,1,1)
        self.std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    def extract(self,path,start=0.,end=None,preview_path=None,roi_cache_path=None):
        rois,quality=self.cropper.extract(path,start,end,preview_path,roi_cache_path)
        images=rois['images'];count=len(images)
        x=torch.from_numpy(images.reshape(-1,224,224,3)).permute(0,3,1,2).float()/255
        x=(x-self.mean)/self.std
        with torch.inference_mode():
            emb=np.concatenate([self.cnn.avgpool(self.cnn.features(b)).flatten(1).numpy() for b in x.split(8)])
        emb=emb.reshape(count,3,576)
        emb[:,0]*=(rois['quality'][:,0]>0)[:,None]
        emb[:,1]*=(rois['quality'][:,1]>0)[:,None]
        combined=np.concatenate([emb.reshape(count,-1),rois['motion']],axis=1)
        features=combined[rois['sample_positions']].astype(np.float32)
        if features.shape!=(STEPS,DIM) or not np.isfinite(features).all():
            raise ValueError('Invalid ROI features')
        return features,quality


class FrameClip:
    """In-memory unmirrored BGR clip; camera timestamps are normalized once."""
    def __init__(self,frames,timestamps):
        self.frames=list(frames);self.times=np.asarray(timestamps,dtype=np.float64)
        if len(self.frames)<4 or self.times.shape!=(len(self.frames),) or not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0):
            raise ValueError('Need four frames and strictly increasing timestamps')
        shape=self.frames[0].shape
        if len(shape)!=3 or shape[2]!=3 or any(f.shape!=shape or f.dtype!=np.uint8 for f in self.frames):
            raise ValueError('Expected same-sized uint8 BGR frames')
    def capture(self):return _MemoryCapture(self)


class _MemoryCapture:
    def __init__(self,clip):
        self.clip=clip;self.position=0;self.n=len(clip.frames)
        targets=np.linspace(clip.times[0],clip.times[-1],self.n)
        right=np.clip(np.searchsorted(clip.times,targets),0,self.n-1);left=np.maximum(right-1,0)
        self.indices=np.where(abs(clip.times[left]-targets)<=abs(clip.times[right]-targets),left,right)
        self.fps=(self.n-1)/(clip.times[-1]-clip.times[0])
    def isOpened(self):return True
    def get(self,prop):
        return self.fps if prop==cv2.CAP_PROP_FPS else self.n if prop==cv2.CAP_PROP_FRAME_COUNT else 0
    def read(self):
        if self.position>=self.n:return False,None
        frame=self.clip.frames[int(self.indices[self.position])];self.position+=1
        return True,frame
    def release(self):pass
