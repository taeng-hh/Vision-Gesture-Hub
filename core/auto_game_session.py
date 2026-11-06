"""Continuous, bounded webcam windows for in-game sign commands."""
from collections import deque
import numpy as np
from core.sign_language_session import MixedLiveSignSession


class AutoGameSession:
    window_seconds = 2.4
    window_lengths = (.8, 1.2, 1.6, 2.4)
    min_seconds = .5
    predict_interval = .2
    cooldown_seconds = .8

    def __init__(self, model, game, assist=None):
        self.model, self.game = model, game
        self.assist=assist
        self.classes = {c['class_id']: c for c in model.classes}
        self.status = '자동 인식 중: 시작 수어 또는 Enter로 출격하세요.'
        self.cancel_recording()

    def cancel_recording(self):
        self.frames = deque(maxlen=360)
        self.video_frames=deque(maxlen=40)
        if self.assist is not None:self.assist.invalidate()
        self.last_time = self.last_hand = self.last_predict = None
        self.cooldown_until = 0.
        self.candidate, self.matches = None, 0
        self.last_comparison = None
        self.next_predict = None
        self.last_match_time=None

    def update(self, frame, now, rgb=None):
        if self.last_time is not None and (now <= self.last_time or now-self.last_time > .6):
            self.cancel_recording()
        self.last_time = now
        if self.game.state == 'paused':
            self.cancel_recording()
            self.candidate, self.matches = None, 0
            self.status = '일시정지 중 · P로 계속'
            return
        if frame is None or not np.any(frame[:42, 2] >= .2):
            self.candidate, self.matches = None, 0
            if frame is None or self.last_hand is None or now-self.last_hand >= .3:
                self.frames.clear()
                self.video_frames.clear()
                if self.assist is not None:self.assist.invalidate()
            self.status = '자동 인식 대기: 어깨와 손을 화면에 보여 주세요.'
            return
        self.last_hand = now
        if now < self.cooldown_until:
            return
        self.frames.append((now, frame.copy()))
        if self.assist is not None and self.assist.enabled and rgb is not None:
            if not self.video_frames or now-self.video_frames[-1][0]>=.08:
                self.video_frames.append((now,rgb.copy(),frame.copy()))
        while self.video_frames and now-self.video_frames[0][0]>self.window_seconds:
            self.video_frames.popleft()
        while self.frames and now-self.frames[0][0] > self.window_seconds:
            self.frames.popleft()
        if len(self.frames) < 4 or now-self.frames[0][0] < self.min_seconds:
            return
        if self.next_predict is not None and now < self.next_predict-1e-6:
            return
        if self.next_predict is None:
            self.next_predict=now+self.predict_interval
        else:
            steps=max(1,int((now-self.next_predict+1e-6)/self.predict_interval)+1)
            self.next_predict+=steps*self.predict_interval
        self.last_predict = now
        times = np.asarray([t for t, _ in self.frames])
        frames = np.asarray([f for _, f in self.frames])
        try:
            if hasattr(self.model,'compare_windows'):
                clips=[];lengths=[]
                for seconds in self.window_lengths:
                    mask=times>=now-seconds-1e-6
                    if mask.sum()<4 or times[mask][-1]-times[mask][0]<self.min_seconds:
                        continue
                    # Match normal webcam resampling: do not invent confidence values.
                    sampler=MixedLiveSignSession(self.model)
                    sampler.frames=list(frames[mask]);sampler.frame_times=list(times[mask])
                    sample=sampler.timed_frames()
                    if any(abs(length-(times[mask][-1]-times[mask][0]))<.02 for length in lengths):
                        continue
                    clips.append(sample);lengths.append(float(times[mask][-1]-times[mask][0]))
                comparisons=self.model.compare_windows(clips)
                # Prefer absolute support, not an inflated game-only score.
                eligible=[i for i,c in enumerate(comparisons) if c['game']['accepted']]
                pool=eligible or list(range(len(comparisons)))
                best_score=max(comparisons[i]['game']['global_score'] for i in pool)
                comparable=[i for i in pool if comparisons[i]['game']['global_score']>=best_score-.05]
                selected=max(comparable,key=lambda i:lengths[i])
                pair=comparisons[selected]
                if self.assist is not None:
                    pair=self.assist.evaluate(pair,list(self.video_frames),now,lengths[selected])
                result=pair['game']
                self.last_comparison=dict(window_seconds=lengths[selected],normal=pair['normal'],
                                          game=result,windows=[dict(seconds=n,**c) for n,c in zip(lengths,comparisons)])
                normal=self.classes.get(pair['normal']['candidate_id'],{}).get('text','?')
                game_name=self.classes.get(result['candidate_id'],{}).get('text','?')
                detail=(f"동일 {lengths[selected]:.1f}초 · 일반 {normal} {pair['normal']['global_score']:.2f} "
                        f"/ 게임 {game_name} 전체 {result['global_score']:.2f}, 비교 {result['score']:.2f}")
            else:
                target = np.linspace(times[0], times[-1], len(times))
                flat = frames.reshape(len(frames), -1)
                clip = np.stack([np.interp(target, times, c) for c in flat.T], axis=1).reshape(frames.shape)
                predictor = getattr(self.model, 'predict_game', self.model.predict)
                result = predictor(clip)
                detail=None
        except ValueError:
            self.frames.clear()
            self.video_frames.clear()
            if self.assist is not None:self.assist.invalidate()
            self.candidate, self.matches = None, 0
            self.status = '좌표 추적 대기: 손과 어깨를 화면에 유지하세요.'
            return
        entry = self.classes.get(result.get('class_id'), {})
        action = entry.get('game_action') if result.get('accepted') else None
        if not action:
            # An asynchronous second check takes longer than one camera tick.
            # Preserve (do not increment) a prior match only while the same
            # candidate has an outstanding CNN check, for a bounded duration.
            waiting=(self.assist is not None and self.assist.enabled and self.assist.pending is not None
                     and self.matches>0 and self.last_match_time is not None
                     and now-self.last_match_time<=1.5
                     and self.classes.get(result.get('candidate_id'),{}).get('game_action')==self.candidate)
            if not waiting:self.candidate,self.matches=None,0
            if result.get('accepted') and entry:
                self.status = f"자동 인식: {entry['text']} · 점수 {result['score']:.2f} · 게임 조작 없음"
            else:
                candidate = self.classes.get(result.get('candidate_id'), {})
                if result.get('scope') == 'game':
                    self.status = (f"게임 후보: {candidate.get('text', '?')} · 비교 {result['score']:.2f} "
                                   f"/ 전체 {result['global_score']:.2f} · 조작 보류")
                else:
                    self.status = '자동 인식 중: 게임 조작 수어를 보여 주세요.'
            if detail is not None:
                self.status=detail+' · 보류'
            return
        self.matches = self.matches + 1 if action == self.candidate else 1
        self.candidate = action
        self.last_match_time=now
        fast=False
        if detail is not None:
            normal=self.last_comparison['normal']
            fast=(normal['accepted'] and normal['candidate_id']==result['candidate_id']
                  and normal['global_score']>=.99 and normal['margin']>=.9
                  and result['global_score']>=.99)
            if not fast and times[-1]-times[0]<1.25:
                self.status=detail+' · 동작 문맥 대기'
                return
        if self.matches < 2 and not fast:
            if detail is not None:self.status=detail+' · 확인 1/2'
            return
        self.game.action(action)
        self.status = (detail+' · 실행') if detail is not None else f"자동 인식: {entry['text']} · 점수 {result['score']:.2f}"
        self.frames.clear()
        self.video_frames.clear()
        if self.assist is not None:self.assist.invalidate()
        self.candidate, self.matches = None, 0
        self.cooldown_until = now + self.cooldown_seconds
