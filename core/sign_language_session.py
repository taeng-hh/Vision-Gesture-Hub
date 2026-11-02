"""UI-independent webcam adapter and word/sentence state."""
from itertools import permutations
import numpy as np

NAMES = {'I': '나', 'HEAD': '머리', 'HURT': '아프다', 'HOSPITAL': '병원',
         'GO': '가다', 'FOOD': '밥', 'REQUEST': '부탁', 'HELP': '돕다'}
RULES = {('I', 'HEAD', 'HURT'): '저는 머리가 아파요.',
         ('HEAD', 'HURT'): '머리가 아파요.',
         ('I', 'HOSPITAL', 'GO'): '저는 병원에 가요.',
         ('HOSPITAL', 'GO'): '병원에 가요.',
         ('FOOD', 'REQUEST'): '밥을 주세요.',
         ('HELP', 'REQUEST'): '도와주세요.'}


def webcam_frame(observation, pose, width, height):
    """Unmirrored MediaPipe coordinates -> OpenPose-like pixel XY/confidence.

    Match hand detections to anatomical pose wrists, not recognizer list order
    or mirror-dependent handedness labels. Z never becomes confidence.
    Returns None on missing shoulders or ambiguous wrist association.
    """
    if pose is None or len(pose) != 33:
        return None
    confidence = lambda p: min(p.visibility or 0., p.presence or 0.)
    if any(confidence(pose[i]) < .5 for i in (11, 12)):
        return None
    scale = np.asarray([width, height], dtype=np.float32)
    xy = lambda p: np.asarray([p.x, p.y], dtype=np.float32) * scale
    body = np.linalg.norm(xy(pose[11]) - xy(pose[12]))
    if body < 20:
        return None
    frame = np.zeros((67, 3), dtype=np.float32)
    for mp_index, op_index in ((11, 5), (12, 2), (15, 7), (16, 4)):
        frame[42 + op_index] = [*xy(pose[mp_index]), confidence(pose[mp_index])]
    hands = [h for h in observation.hands if len(h.landmarks) == 21]
    if not hands:
        return frame
    if len(hands) > 2:
        return None
    costs = []
    for slots in permutations(range(2), len(hands)):
        distances = [np.linalg.norm(xy(h.landmarks[0]) - xy(pose[(15, 16)[slot]])) / body
                     for h, slot in zip(hands, slots)]
        if all(confidence(pose[(15, 16)[s]]) >= .5 for s in slots):
            costs.append((sum(distances), slots, distances))
    if not costs:
        return None
    costs.sort()
    if max(costs[0][2]) > .65 or (len(costs) > 1 and costs[1][0] - costs[0][0] < .10):
        return None
    for hand, slot in zip(hands, costs[0][1]):
        for i, point in enumerate(hand.landmarks):
            frame[slot * 21 + i] = [*xy(point), 1.]
    return frame


class LiveSignSession:
    """Explicit recording boundary; no forced predictions on idle frames."""
    def __init__(self, model):
        self.model = model
        self.max_seconds = 6
        self.max_frames = 360
        self.tokens = []
        self.sentence = ''
        self.status = 'Space를 누르고 단어 한 개를 수행하세요.'
        self.cancel_recording()

    def cancel_recording(self):
        self.recording = False
        self.frames = []
        self.started = self.last_hand = None
        self.gap_since = None
        self.last_time = None

    def start(self, now):
        self.cancel_recording()
        self.recording = True
        self.started = now
        self.status = '녹화 중: 단어 수행 후 Space 또는 손을 화면 밖으로 빼세요.'

    def finish_word(self):
        frames = self.frames
        duration = (self.last_hand - self.started
                    if self.last_hand is not None and self.started is not None else 0)
        self.cancel_recording()
        if len(frames) < 4 or duration < .20:
            self.status = '동작이 너무 짧습니다. Space로 다시 시작하세요.'
            return False
        try:
            result = self.model.predict(frames)
        except ValueError:
            self.status = '유효한 손/어깨 좌표가 부족합니다. 다시 시도하세요.'
            return False
        label = result['label']
        detail = f"거리 {result['distance']:.2f}, 차이 {result['margin']:.2f}"
        if label is None:
            self.status = f'인식 불가 ({detail}). Space로 다시 시도하세요.'
            return False
        if len(self.tokens) >= 12:
            self.status = '단어가 너무 많습니다. C로 초기화하세요.'
            return False
        self.tokens.append(label)
        # Exact supported combinations automatically produce a sentence preview.
        self.sentence = RULES.get(tuple(self.tokens), '')
        self.status = f'인식: {NAMES.get(label, label)} ({detail})'
        return True

    def update(self, frame, now):
        if not self.recording:
            return
        if frame is None:
            self.cancel_recording()
            self.status = '어깨/손 연결 추적 실패: 취소했습니다. 상반신을 보이게 하세요.'
            return
        if self.last_time is not None and (now - self.last_time > .75 or now < self.last_time):
            self.cancel_recording()
            self.status = '입력이 끊겨 녹화를 취소했습니다. 다시 시작하세요.'
            return
        self.last_time = now
        if now - self.started > self.max_seconds or len(self.frames) >= self.max_frames:
            self.cancel_recording()
            self.status = f'{self.max_seconds}초 제한을 넘겨 취소했습니다. 다시 입력하세요.'
            return
        if np.any(frame[:42, 2] >= .2):
            if not self.frames:
                self.started = now
            self.frames.append(frame.copy())
            self.last_hand = now
            self.gap_since = None
        elif self.frames:
            if self.gap_since is None:
                self.gap_since = now
            elif now - self.gap_since >= .35:
                self.finish_word()

    def handle_key(self, key, now):
        if key == ord(' '):
            self.finish_word() if self.recording else self.start(now)
        elif key in (10, 13):
            if self.recording and not self.finish_word():
                return
            self.sentence = RULES.get(tuple(self.tokens), '지원하지 않는 단어 조합입니다.')
        elif key in (8, 127):
            self.cancel_recording()
            if self.tokens: self.tokens.pop()
            self.sentence = RULES.get(tuple(self.tokens), '')
        elif key in (ord('c'), ord('C')):
            self.cancel_recording()
            self.tokens = []
            self.sentence = ''
            self.status = '초기화했습니다. Space로 시작하세요.'

    @property
    def display_lines(self):
        return ['실험용 8단어 모델 | 새 사용자 정확도 미검증',
                'Space: 단어 시작/완료 | Enter: 문장 | Backspace: 삭제 | C: 초기화',
                self.status, '단어: ' + ' · '.join(NAMES.get(t, t) for t in self.tokens),
                '문장: ' + self.sentence]


class MixedLiveSignSession(LiveSignSession):
    """One recording yields either a word event or a complete sentence."""
    def __init__(self, model):
        super().__init__(model)
        self.names = {c['class_id']: c['text'] for c in model.classes}
        self.max_seconds = 12
        self.max_frames = 1200
        self.status = 'Space로 시작하고 단어 또는 문장 하나를 수행하세요.'

    def start(self, now):
        super().start(now)
        self.status = '입력 중: 단어/문장 하나를 끝낸 뒤 Space를 누르세요.'

    def cancel_recording(self):
        super().cancel_recording()
        self.frame_times = []
        self.tracking_lost_since = None

    def update(self, frame, now):
        # Explicit end only: short hand occlusions should not finish a sentence.
        if not self.recording:
            return
        if now < self.started or now - self.started > self.max_seconds:
            self.cancel_recording()
            self.status = '입력 시간 제한: Space로 다시 시작하세요.'
            return
        if self.last_time is not None and (now < self.last_time or now-self.last_time > 1.5):
            self.cancel_recording()
            self.status = '카메라 입력이 끊겨 취소했습니다.'
            return
        self.last_time = now
        if frame is None:
            if self.tracking_lost_since is None:
                self.tracking_lost_since = now
            if now-self.tracking_lost_since >= .6:
                self.cancel_recording()
                self.status = '0.6초 이상 손/어깨 추적 실패: 상반신을 보이게 해 주세요.'
            else:
                self.status = '잠깐 추적 놓침: 손과 어깨를 화면에 유지하세요.'
            return
        self.tracking_lost_since = None
        has_hand = np.any(frame[:42, 2] >= .2)
        if not self.frames and not has_hand:
            self.status = '입력 대기: 손을 보이고 동작 후 Space로 완료하세요.'
            return
        if len(self.frames) >= self.max_frames:
            self.cancel_recording()
            self.status = '입력 프레임 제한: 다시 시작하세요.'
            return
        self.frames.append(frame.copy())
        self.frame_times.append(now)
        if has_hand:
            self.last_hand = now
        self.status = f'입력 중 {now-self.started:.1f}초 / {len(self.frames)}프레임 · Space로 완료'

    def timed_frames(self):
        # Camera FPS varies. Resample actual timestamps instead of assuming equal spacing.
        frames = np.asarray(self.frames, dtype=np.float32)
        if len(self.frame_times) != len(frames) or len(frames) < 4:
            return frames
        times = np.asarray(self.frame_times)
        if np.any(np.diff(times) <= 0):
            raise ValueError('Non-increasing timestamps')
        target = np.linspace(times[0], times[-1], len(frames))
        flat = frames.reshape(len(frames), -1)
        return np.stack([np.interp(target, times, c) for c in flat.T], axis=1).reshape(frames.shape)

    def compose(self):
        rules = {('WORD1534', 'WORD1589'): '밥을 주세요.',
                 ('WORD0115', 'WORD1589'): '커피를 주세요.',
                 ('WORD1157', 'WORD0953'): '저는 배가 고파요.',
                 ('WORD1157', 'WORD1169'): '저는 행복해요.',
                 ('WORD1157', 'WORD0009'): '저는 슬퍼요.',
                 ('WORD0003', 'WORD1191'): '수어가 좋아요.'}
        return rules.get(tuple(self.tokens), ' '.join(self.names[t] for t in self.tokens))

    def finish_word(self):
        frames = self.timed_frames()
        duration = self.last_hand - self.started if self.last_hand is not None else 0
        self.cancel_recording()
        if len(frames) < 4 or duration < .20:
            self.status = '동작이 너무 짧습니다. Space로 다시 시작하세요.'
            return False
        visible = np.any(np.asarray(frames)[:, :42, 2] >= .2, axis=1)
        if np.mean(visible) < .6:
            self.status = '손이 보이는 프레임이 부족합니다. 손을 화면 안에 유지하세요.'
            return False
        try:
            result = self.model.predict(frames)
        except ValueError:
            self.status = '유효한 손/어깨 좌표가 부족합니다. 다시 시도하세요.'
            return False
        if not result['accepted']:
            candidate = self.names.get(result.get('candidate_id'), '알 수 없음')
            self.status = f"인식 보류: 후보 {candidate}, 점수 {result['score']:.2f}, 차이 {result.get('margin', 0):.2f}"
            return False
        if result['kind'] == 'sentence':
            self.tokens = []
            self.sentence = result['text']
            self.status = f"문장 인식: {result['text']} (점수 {result['score']:.2f})"
        else:
            if len(self.tokens) >= 12:
                self.status = '단어가 너무 많습니다. C로 초기화하세요.'
                return False
            self.tokens.append(result['class_id'])
            self.sentence = self.compose()
            self.status = f"단어 인식: {result['text']} (점수 {result['score']:.2f})"
        return True

    def handle_key(self, key, now):
        if key in (10, 13):
            if self.recording and not self.finish_word():
                return
            if self.tokens:
                self.sentence = self.compose()
        elif key in (8, 127):
            self.cancel_recording()
            if self.tokens:
                self.tokens.pop()
            self.sentence = self.compose()
        else:
            super().handle_key(key, now)

    @property
    def display_lines(self):
        return ['30클래스: 단어 23 + 문장 7 | 실험용',
                'Space: 시작/완료 | Enter: 조합 | Backspace: 삭제 | C: 초기화',
                self.status,
                '단어: ' + ' · '.join(self.names[t] for t in self.tokens),
                '출력: ' + self.sentence]
