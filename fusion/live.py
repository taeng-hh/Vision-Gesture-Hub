"""Explicit, bounded in-memory recording with asynchronous fusion inference."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import cv2
import numpy as np
from .preview import PreviewTracker,draw_skeleton,tracking_state
from ui import theme
from ui.translator_view import TranslatorState,draw_translator
from core.sign_language_session import MixedLiveSignSession
from .shared import FrameClip,SharedExtractor
from .predict import select_prediction,MODE_LABELS
from .fast import analyze_clip,needs_mode_analysis


class ClipRecorder:
    def __init__(self,max_seconds=12,max_bytes=1024*1024*1024,max_fps=30):
        self.max_seconds=max_seconds;self.max_bytes=max_bytes;self.max_fps=max_fps
        self.cancel()
    def cancel(self):
        self.recording=False;self.frames=[];self.times=[];self.bytes=0;self.started=None;self.last_input=None;self.next_capture=None;self.landmark_samples={}
    def start(self,now):
        self.cancel();self.recording=True;self.started=now
    def update(self,frame,now,snapshot=None):
        if not self.recording:return
        if now<self.started or now-self.started>self.max_seconds or (self.last_input is not None and now-self.last_input>1.5):
            self.cancel();raise ValueError("입력 시간 제한 또는 카메라 끊김: 다시 시작하세요.")
        self.last_input=now
        if snapshot is not None and getattr(snapshot,'keypoints',None) is not None and self.started<=snapshot.captured_at<=now:
            if snapshot.captured_at not in self.landmark_samples:self.landmark_samples[snapshot.captured_at]=snapshot.keypoints.copy()
        if self.next_capture is not None and now+.002<self.next_capture:return
        if self.bytes+frame.nbytes>self.max_bytes:
            self.cancel();raise ValueError("입력 메모리 제한: 짧게 수행한 뒤 다시 시도하세요.")
        self.frames.append(frame.copy());self.times.append(now);self.bytes+=frame.nbytes
        self.next_capture=max((self.next_capture if self.next_capture is not None else now)+1/self.max_fps,now+.5/self.max_fps)
    def finish(self):
        frames,times=self.frames,self.times
        samples=tuple(sorted(self.landmark_samples.items()))
        self.cancel()
        if len(frames)<4 or times[-1]-times[0]<.2:raise ValueError("동작이 너무 짧습니다.")
        clip=FrameClip(frames,times);clip.landmark_samples=samples
        return clip


CAMERA_GRACE_SECONDS=2.5  # Consecutive failed reads before giving up.


def run_camera(model,assets,camera=0,mode="assisted",reuse_live=True,game=None):
    """game: optional fusion.game_switch.GameSwitch (V hand shape -> Galaga)."""
    if mode not in MODE_LABELS:raise ValueError("Unknown recognition mode")
    cropper=SharedExtractor(assets);recorder=ClipRecorder()
    executor=ThreadPoolExecutor(max_workers=1);future=None;generation=0;future_generation=0
    tokens=[];sentence="";components="";quality_text="";message="Space로 시작하고 단어 또는 문장 하나를 수행하세요."
    last_result=None;preview_rois=None;last_clip=None;future_append=True
    event=None;event_at=None;debug=False;recent=[];history=[]
    last_display=None;fail_since=None;reopened=False
    names={c["class_id"]:c["text"] for c in model.classes}
    compose=type("Composition",(),{})();compose.tokens=tokens;compose.names=names
    def view_state(display,now,snapshot,status,**extra):
        """Everything the screen shows, read from the loop's current values."""
        debug_lines=("판정 모드: "+MODE_LABELS[mode]+" / 30클래스",
                     f"출력 기준: 점수 {getattr(model,'min_score',.75):.2f} / 차이 {getattr(model,'min_margin',.15):.2f}",
                     *(components.split(" | ") if components else ()),*((quality_text,) if quality_text else ()),
                     *((f"입력 {len(recorder.frames)}프레임",) if recorder.recording else ()))
        return TranslatorState(camera=display,now=now,status=status,
                               elapsed=now-recorder.started if recorder.recording else 0.,record_limit=recorder.max_seconds,
                               event=event,event_age=None if event_at is None else now-event_at,message=message,
                               tokens=tuple(names[t] for t in tokens),sentence=sentence,history=tuple(history),recent=tuple(recent),
                               **tracking_state(snapshot,now),
                               game_progress=None if game is None or game.error else game.progress,
                               game_note=(game.error or "") if game is not None else "게임 전환 꺼짐 (--no-game)",
                               debug=debug,debug_lines=debug_lines,rois=preview_rois,**extra)
    def analyze(clip,selected_mode):
        return analyze_clip(model,assets,clip,selected_mode,reuse_live=reuse_live)
    cap=cv2.VideoCapture(camera);tracker=None
    try:
        if not cap.isOpened():raise RuntimeError("카메라를 열 수 없습니다. --camera 1을 시도하세요.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        tracker=PreviewTracker(assets)
        window=theme.WINDOW_TITLE  # Shared with the game, which draws into the same window.
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window,theme.WIDTH,theme.HEIGHT)
        while True:
            ok,raw=cap.read()
            now=time.monotonic()
            if not ok:
                # Webcams drop frames briefly (start-up, other apps); only give up after a grace period.
                if fail_since is None:fail_since=now;reopened=False
                if recorder.recording:
                    recorder.cancel();tracker.set_max_fps(12);message="카메라 입력이 끊겨 녹화를 취소했습니다. Space로 다시 시작하세요."
                if now-fail_since>CAMERA_GRACE_SECONDS:raise RuntimeError("카메라 입력이 중단됐습니다.")
                if now-fail_since>1. and not reopened:
                    reopened=True;cap.release()
                    if cap.open(camera):cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
                cv2.imshow(window,draw_translator(view_state(last_display,now,None,"reconnecting",camera_note="카메라 다시 연결 중…")))
                key=cv2.waitKey(30)&0xff
                if key in (ord("q"),ord("Q"),27):break
                continue
            fail_since=None
            tracker.submit(raw,now)
            snapshot=tracker.snapshot()
            try:recorder.update(raw,now,snapshot)
            except ValueError as exc:tracker.set_max_fps(12);message=str(exc)
            if game is not None and game.update(raw,now,recording=recorder.recording):
                # Short hand-off card, then the game opens its own capture in the same window.
                cv2.imshow(window,draw_translator(view_state(last_display if last_display is not None else cv2.flip(raw,1),now,snapshot,"ready",
                                                             overlay=("SIGN GALAGA로 이동합니다","게임에서 Q 또는 Esc를 누르면 번역 화면으로 돌아옵니다"))))
                cv2.waitKey(400)
                cap.release()
                message=game.launch(window=window)
                if not cap.open(camera):raise RuntimeError("게임 후 카메라를 다시 열 수 없습니다.")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
                try:closed=cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1
                except cv2.error:closed=True
                if closed:  # Closing the game window returns to the translator, as before.
                    cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,theme.WIDTH,theme.HEIGHT)
                continue
            if future is not None and future.done() and future_generation!=generation:
                future=None
            if future is not None and future.done():
                try:
                    new_result=future.result()
                    if not future_append and last_result is not None:
                        for field in ('keypoint_probabilities','rgb_probabilities'):
                            if new_result.get(field) is None:new_result[field]=last_result.get(field)
                        if new_result.get('preview_rois') is None:new_result['preview_rois']=last_result.get('preview_rois')
                        new_result['computed_modalities']=list(set(new_result.get('computed_modalities',()))|set(last_result.get('computed_modalities',())))
                        new_result['available_modalities']=[name for name in ('keypoint','rgb') if new_result.get(name+'_probabilities') is not None]
                    last_result=new_result
                    preview_rois=last_result["preview_rois"]
                    quality=last_result["input_quality"]
                    quality_text=("추적 재사용 · " if last_result["pipeline"]=="live_keypoints" else "원본 분석 · ")+f"직접 손 검출: L {quality['left_direct_detection_fraction']:.0%} / R {quality['right_direct_detection_fraction']:.0%}"
                    parts=[]
                    for name,label in (("keypoint_probabilities","키포인트"),("rgb_probabilities","영상")):
                        scores=last_result.get(name)
                        if scores is not None:
                            i=max(range(len(scores)),key=scores.__getitem__)
                            parts.append(f"{label}: {names[model.ids[i]]} {scores[i]:.2f}")
                        else:parts.append(label+(": 입력 부족" if name.removesuffix("_probabilities") in last_result.get("component_errors",{}) else ": 미계산"))
                    components=" | ".join(parts)
                    result=select_prediction(last_result,model.classes,mode,model.min_score,model.min_margin)
                    if mode=="assisted":
                        status_labels={"keypoint_confident":"키포인트 확정 · CNN 미실행","cnn_confirmed":"CNN 후보 일치 확인","cnn_disagrees":"CNN 후보 불일치 · 보류","cnn_uncertain":"CNN도 불확실 · 보류","still_uncertain":"두 모델 확인 후에도 보류","cnn_unavailable":"CNN 확인 불가 · 보류"}
                        quality_text=status_labels.get(result.get("assist_status"),"키포인트 우선 판정")
                    event=dict(accepted=result["accepted"],kind=result["kind"],score=result["score"],
                               text=result["text"] if result["accepted"] else names.get(result["candidate_id"],""),
                               candidates=[(a["text"],a["score"]) for a in result.get("alternatives",())])
                    if result["accepted"]:
                        if not future_append:pass
                        else:
                            event_at=now;recent.insert(0,(result["text"],result["score"]));del recent[3:]
                            if result["kind"]=="sentence":
                                if sentence and sentence!=result["text"]:history.append(sentence);del history[:-2]
                                tokens.clear();sentence=result["text"]
                            elif len(tokens)<12:
                                tokens.append(result["class_id"]);sentence=MixedLiveSignSession.compose(compose)
                        message=f"{MODE_LABELS[mode]}: {result['text']} · {result['score']:.2f} · 분석 {last_result['analysis_seconds']:.2f}초"
                    else:message=f"{MODE_LABELS[mode]} 인식 보류: {names.get(result['candidate_id'],'입력 없음')} · 점수 {result['score']:.2f} · 차이 {result['margin']:.2f}"
                except (ValueError,RuntimeError) as exc:message="분석 실패: "+str(exc)
                future=None
            snapshot=tracker.snapshot()
            display=cv2.flip(raw,1)
            draw_skeleton(display,snapshot,now,mirror=True)
            last_display=display
            status="recording" if recorder.recording else "analyzing" if future else "ready"
            cv2.imshow(window,draw_translator(view_state(display,now,snapshot,status)))
            key=cv2.waitKey(1)&0xff
            if key in (ord("q"),ord("Q"),27):break
            if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key in (ord("1"),ord("2"),ord("3")):
                if future is not None or recorder.recording:
                    message="입력·분석이 끝난 뒤 모드를 바꿔주세요.";continue
                mode={ord("1"):"keypoint",ord("2"):"rgb",ord("3"):"assisted"}[key]
                if last_result is not None:
                    try:
                        if needs_mode_analysis(last_result,mode,model.min_score,model.min_margin):raise ValueError("추가 모델 계산 필요")
                        compared=select_prediction(last_result,model.classes,mode,model.min_score,model.min_margin)
                        message=f"최근 동작 · {MODE_LABELS[mode]}: {names[compared['candidate_id']]} · {compared['score']:.2f} · "+("허용" if compared["accepted"] else "보류")
                    except ValueError:
                        if last_clip is not None:
                            future=executor.submit(analyze,last_clip,mode);future_generation=generation;future_append=False
                            message=MODE_LABELS[mode]+" 모드를 추가 분석합니다."
                        else:message="새 동작을 입력해주세요."
                else:message=MODE_LABELS[mode]+" 판정 모드로 변경했습니다."
            elif key==ord(" "):
                if future is not None:message="분석 중입니다. 결과가 나온 뒤 다시 시작하세요."
                elif recorder.recording:
                    try:
                        last_clip=recorder.finish();tracker.set_max_fps(12)
                        future=executor.submit(analyze,last_clip,mode);future_generation=generation;future_append=True;message=MODE_LABELS[mode]+" 분석 중입니다."
                    except ValueError as exc:tracker.set_max_fps(12);message=str(exc)
                else:
                    last_clip=None;last_result=None;preview_rois=None;components=""
                    tracker.set_max_fps(30);recorder.start(now);message="입력 중: 동작을 마치면 Space를 누르세요."
            elif key in (ord("d"),ord("D")):debug=not debug
            elif key in (ord("c"),ord("C")):
                # First C keeps the finished sentence above the subtitle; C on an empty subtitle clears everything.
                if sentence:history.append(sentence);del history[:-2];message="다음 문장을 시작합니다."
                else:history.clear();recent.clear();message="초기화했습니다."
                generation+=1;recorder.cancel();tokens.clear();sentence="";components="";quality_text="";last_result=None;preview_rois=None;last_clip=None;future_append=True;tracker.set_max_fps(12);event=None;event_at=None
            elif key in (8,127):
                generation+=1;recorder.cancel();tracker.set_max_fps(12)
                if tokens:tokens.pop()
                sentence=MixedLiveSignSession.compose(compose) if tokens else ""
    finally:
        recorder.cancel();cap.release()
        if tracker is not None:tracker.close()
        cv2.destroyAllWindows()
        executor.shutdown(wait=False,cancel_futures=True)
