"""Asynchronous diagnostic landmarks, drawn only on the camera preview."""
from dataclasses import dataclass
from pathlib import Path
from threading import Condition,Thread
from types import SimpleNamespace
import time
import cv2
import numpy as np
from core.sign_language_session import webcam_frame
from ui.ksl_overlay import get_font

HAND_EDGES=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17))
BODY_EDGES=((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24))
BODY_POINTS=(11,12,13,14,15,16,23,24)
HAND_COLORS=((40,230,80),(0,220,255))
BODY_COLOR=(255,200,30)

@dataclass(frozen=True)
class TrackingSnapshot:
    captured_at:float
    hands:tuple=()
    pose:tuple=()
    shoulders_ok:bool=False
    keypoint_ok:bool=False
    error:str=''
    keypoints:object=None


def snapshot_from_results(hands,pose,width,height,captured_at):
    pp=pose.pose_landmarks[0] if pose.pose_landmarks else None
    hs=hands.hand_landmarks
    confidence=lambda p:min(float(p.visibility or 0),float(p.presence or 0))
    body=tuple((float(p.x),float(p.y),confidence(p)) for p in pp) if pp else ()
    fingers=tuple(tuple((float(p.x),float(p.y)) for p in hand) for hand in hs)
    shoulders=bool(pp and min(confidence(pp[11]),confidence(pp[12]))>=.5 and np.linalg.norm(np.array([(pp[11].x-pp[12].x)*width,(pp[11].y-pp[12].y)*height]))>=20)
    frame=webcam_frame(SimpleNamespace(hands=[SimpleNamespace(landmarks=h) for h in hs]),pp,width,height)
    # The shared adapter keeps shoulders when detected hands cannot be matched to
    # pose wrists. Live landmark reuse skips such frames instead of recording them
    # as "no hands" (the behaviour this engine was validated with).
    if frame is not None and hs and not np.any(frame[:42,2]>0):frame=None
    usable=False
    if frame is not None:
        for hand in (frame[:21],frame[21:42]):
            valid=hand[:,2]>=.2
            if valid[0] and valid[9] and valid.sum()>=15 and np.linalg.norm(hand[9,:2]-hand[0,:2])>1e-6:usable=True
    return TrackingSnapshot(captured_at,fingers,body,shoulders,usable,keypoints=frame)


class PreviewTracker:
    """Keep one pending frame; never block camera capture on MediaPipe."""
    def __init__(self,assets,max_fps=12):
        self.assets=Path(assets);self.period=1/max_fps
        self.condition=Condition();self.pending=None;self.latest=None;self.closed=False;self.last_submit=-float('inf');self.next_submit=None
        self.thread=Thread(target=self._run,name='ksl-preview-tracking',daemon=True);self.thread.start()
    def set_max_fps(self,value):
        with self.condition:self.period=1/value;self.next_submit=None
    def submit(self,frame,captured_at):
        with self.condition:
            if self.closed or (self.next_submit is not None and captured_at+.002<self.next_submit):return
            self.last_submit=captured_at
            self.next_submit=max((self.next_submit if self.next_submit is not None else captured_at)+self.period,captured_at+.5*self.period)
            self.pending=(frame.copy(),captured_at)
            self.condition.notify()
    def snapshot(self):
        with self.condition:return self.latest
    def close(self):
        with self.condition:
            self.closed=True;self.pending=None;self.condition.notify()
        self.thread.join(timeout=2)
    def _run(self):
        from contextlib import ExitStack
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            with ExitStack() as stack:
                hand=stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(self.assets/'hand_landmarker.task')),running_mode=vision.RunningMode.VIDEO,num_hands=2)))
                pose=stack.enter_context(vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(self.assets/'pose_landmarker_lite.task')),running_mode=vision.RunningMode.VIDEO,num_poses=1)))
                previous_ms=-1
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda:self.closed or self.pending is not None)
                        if self.closed:return
                        frame,captured_at=self.pending;self.pending=None
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                    stamp=max(previous_ms+1,round(captured_at*1000));previous_ms=stamp
                    h=hand.detect_for_video(image,stamp);p=pose.detect_for_video(image,stamp)
                    result=snapshot_from_results(h,p,frame.shape[1],frame.shape[0],captured_at)
                    with self.condition:self.latest=result
        except Exception as exc:
            with self.condition:self.latest=TrackingSnapshot(time.monotonic(),error=str(exc));self.closed=True;self.pending=None


def tracking_status(snapshot,now):
    if snapshot is None:return ['손/상체 추적 준비 중...']
    if snapshot.error:return ['추적 오류: '+snapshot.error]
    if now-snapshot.captured_at>.75:return ['추적 갱신 대기 · 현재 위치 미확인']
    lines=[f'손 검출: {len(snapshot.hands)}/2', '양어깨: '+('확인됨' if snapshot.shoulders_ok else '미검출 — 상반신을 보여주세요')]
    if snapshot.keypoint_ok:lines.append('키포인트 입력: 사용 가능')
    elif not snapshot.hands:lines.append('손을 카메라에 보여주세요')
    else:lines.append('손과 양어깨 위치를 함께 확인하세요')
    return lines


def draw_skeleton(frame,snapshot,now,mirror=True):
    """Frame is the mirrored display copy; raw recording is never modified."""
    if snapshot is None or snapshot.error or now-snapshot.captured_at>.75:return frame
    height,width=frame.shape[:2]
    def point(p):
        x,y=p[:2]
        if not np.isfinite([x,y]).all() or not (0<=x<=1 and 0<=y<=1):return None
        return round((1-x if mirror else x)*(width-1)),round(y*(height-1))
    def draw(points,edges,color,confidence=False):
        coords=[point(p) if not confidence or p[2]>=.5 else None for p in points]
        for a,b in edges:
            if a<len(coords) and b<len(coords) and coords[a] is not None and coords[b] is not None:cv2.line(frame,coords[a],coords[b],color,2,cv2.LINE_AA)
        selected=BODY_POINTS if confidence else range(len(points))
        for i in selected:
            if i<len(coords) and coords[i] is not None:cv2.circle(frame,coords[i],4 if confidence else 3,color,-1,cv2.LINE_AA)
    if snapshot.pose:draw(snapshot.pose,BODY_EDGES,BODY_COLOR,True)
    for i,hand in enumerate(snapshot.hands):draw(hand,HAND_EDGES,HAND_COLORS[i%2])
    return frame


def diagnostic_view(preview,lines,rois=None):
    """Keep all diagnostic text beside the image, so it cannot hide hands."""
    from PIL import Image,ImageDraw
    h,w=preview.shape[:2];scale=min(900/w,600/h,1.)
    image=cv2.resize(preview,(max(1,round(w*scale)),max(1,round(h*scale)))) if scale<1 else preview
    has_rois=rois is not None and np.asarray(rois).shape==(3,224,224,3)
    height=max(600,image.shape[0]+(152 if has_rois else 0));canvas=np.full((height,image.shape[1]+380,3),22,np.uint8)
    canvas[:image.shape[0],:image.shape[1]]=image
    if has_rois:
        size=min(128,max(1,(image.shape[1]-24)//3));top=image.shape[0]+24
        for i,(roi,label) in enumerate(zip(rois,['Left hand','Right hand','Upper body'])):
            left=8+i*(size+8)
            cv2.putText(canvas,label,(left,top-7),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1,cv2.LINE_AA)
            canvas[top:top+size,left:left+size]=cv2.resize(roi[:,:,::-1],(size,size))
    panel=Image.new('RGB',(380,height),(22,22,22));draw=ImageDraw.Draw(panel);font=get_font();y=10
    for line in lines:
        wrapped=''
        for c in line:
            if wrapped and draw.textlength(wrapped+c,font=font)>356:
                draw.text((12,y),wrapped,font=font,fill=(245,245,245));y+=24;wrapped=''
            wrapped+=c
        draw.text((12,y),wrapped,font=font,fill=(245,245,245));y+=24
        if y>=height-24:break
    canvas[:,image.shape[1]:]=np.asarray(panel)[:,:,::-1]
    return canvas
