"""Reuse recorded live landmarks and execute only the requested model branch."""
import math,time
import numpy as np
from ksl.data import encode
from .shared import SharedExtractor
from .predict import select_prediction


def live_keypoints(clip):
    samples=[(float(t),np.asarray(frame,dtype=np.float32)) for t,frame in getattr(clip,'landmark_samples',()) if clip.times[0]<=t<=clip.times[-1]]
    samples=sorted(dict(samples).items());duration=float(clip.times[-1]-clip.times[0])
    if len(samples)<max(4,math.ceil(duration*12)):raise ValueError('촬영 중 추적 프레임 부족')
    times=np.array([s[0] for s in samples]);frames=np.array([s[1] for s in samples])
    if frames.shape!=(len(times),67,3) or not np.isfinite(frames).all():raise ValueError('추적 관절 형식 오류')
    if times[0]-clip.times[0]>.2 or clip.times[-1]-times[-1]>.2 or np.max(np.diff(times))>.2:raise ValueError('추적 구간에 긴 공백이 있습니다')
    targets=np.linspace(clip.times[0],clip.times[-1],64)
    right=np.clip(np.searchsorted(times,targets),0,len(times)-1);left=np.maximum(right-1,0)
    nearest=np.where(abs(times[left]-targets)<=abs(times[right]-targets),left,right)
    result=frames[nearest]
    encode(result) # Missing/invalid tracking must use the original offline path instead.
    return result


def keypoint_quality(frames):
    left=np.sum(frames[:,:21,2]>=.2,axis=1)>=15
    right=np.sum(frames[:,21:42,2]>=.2,axis=1)>=15
    return dict(left_direct_detection_fraction=float(np.mean(left)),right_direct_detection_fraction=float(np.mean(right)),any_hand_roi_fraction=None)


def analyze_clip(model,assets,clip,mode='assisted',reuse_live=True):
    started=time.perf_counter();live=None;reason=None;rois=None
    try:
        if not reuse_live:raise ValueError("실시간 추적 재사용을 사용하지 않는 설정")
        live=live_keypoints(clip)
    except ValueError as exc:reason=str(exc)
    primary_mode='keypoint' if mode=='assisted' else mode
    if primary_mode=='keypoint' and live is not None:
        result=model.predict_rois({},keypoint_frames=live,mode='keypoint')
        quality=keypoint_quality(live);preview=None;pipeline='live_keypoints'
    else:
        rois,quality=SharedExtractor(assets).extract(clip)
        result=model.predict_rois(rois,keypoint_frames=live,mode=primary_mode)
        preview=rois['images'][len(rois['images'])//2].copy()
        pipeline='offline_keypoint_fallback' if primary_mode=='keypoint' else mode+'_rois'
    if mode=='assisted':
        result=select_prediction(result,model.classes,'assisted',model.min_score,model.min_margin)
        if result['assistance_needed']:
            try:
                if rois is None:
                    rois,quality=SharedExtractor(assets).extract(clip)
                    preview=rois['images'][len(rois['images'])//2].copy()
                auxiliary=model.predict_rois(rois,mode='rgb')
                result['rgb_probabilities']=auxiliary.get('rgb_probabilities')
                result['component_errors'].update(auxiliary.get('component_errors',{}))
            except (ValueError,RuntimeError,OSError) as exc:
                result['component_errors']['rgb']=str(exc)
            result['computed_modalities']=list(dict.fromkeys(result.get('computed_modalities',[])+['rgb']))
            result['available_modalities']=[m for m in ('keypoint','rgb') if result.get(m+'_probabilities') is not None]
            result=select_prediction(result,model.classes,'assisted',model.min_score,model.min_margin)
            pipeline='assisted_cnn'
    result.update(input_quality=quality,preview_rois=preview,pipeline=pipeline,
                  fast_path_fallback_reason=reason,analysis_seconds=time.perf_counter()-started)
    return result


def needs_mode_analysis(result,mode,min_score=.75,min_margin=.15):
    computed=set(result.get('computed_modalities',result.get('available_modalities',())))
    if mode=='assisted':
        if 'keypoint' not in computed:return True
        values=result.get('keypoint_probabilities')
        if values is None:return False
        top=np.sort(np.asarray(values))[-2:]
        confident=top[-1]>=min_score and top[-1]-top[-2]>=min_margin
        return not confident and 'rgb' not in computed
    required={"keypoint":{"keypoint"},"rgb":{"rgb"},"fusion":{"keypoint","rgb"}}[mode]
    return not required.issubset(computed)
