"""Recorded KSL sequences -> accepted words -> explicit sentence rules."""
import json
import time
from pathlib import Path
from core.ksl import ClipSegmenter, SentenceComposer, SequenceModel, hand_frame

ROOT = Path(__file__).resolve().parents[1]


class SignLanguageMode:
    def __init__(self, model_path=None, rules_path=None):
        config = json.loads(Path(rules_path or ROOT / "config/ksl_rules.json").read_text(encoding="utf-8"))
        self.composer = SentenceComposer(config["rules"])
        self.names = config.get("display_names", {})
        self.segmenter = ClipSegmenter()
        path = Path(model_path or ROOT / "models/ksl_words.npz")
        self.model = None
        self.error = "학습 모델 없음: 먼저 단어 데이터를 수집하고 학습하세요."
        if path.exists():
            try:
                self.model = SequenceModel.load(path)
            except (ValueError, KeyError, OSError) as exc:
                self.error = f"모델 로드 실패: {exc}"
        self.last_result = ""

    def reset(self):
        self.segmenter.reset()

    def handle_key(self, key):
        if key in (10, 13) and self.composer.tokens:
            self.composer.finish()
        elif key in (8, 127):
            if self.composer.tokens:
                self.composer.tokens.pop()
            self.composer.text = ""
        elif key == ord("c"):
            self.composer.clear()
            self.last_result = ""
        else:
            return
        self.segmenter.reset()

    def update(self, observation, now=None):
        if self.model is None:
            return "KSL model missing or invalid"
        clip = self.segmenter.update(hand_frame(observation), time.monotonic() if now is None else now)
        if clip is not None:
            try:
                result = self.model.predict(clip)
                if result["label"] is not None:
                    added = self.composer.add(result["label"])
                    self.last_result = (f"인식: {self.names.get(result['label'], result['label'])}"
                                        if added else "단어가 너무 많습니다. 문장을 지워 주세요.")
                else:
                    self.last_result = "인식 불가: 동작을 다시 입력하세요."
            except ValueError:
                self.last_result = "유효하지 않은 동작입니다."
        return "SIGN: remove hands between words | Enter: sentence | C: clear"

    @property
    def display_lines(self):
        if self.model is None:
            return [self.error]
        words = " · ".join(self.names.get(t, t) for t in self.composer.tokens)
        return ["단어마다 손을 화면 밖으로 빼 주세요 (0.35초 이상).",
                "Enter: 문장 확정 / Backspace: 단어 삭제 / C: 초기화",
                self.last_result, f"단어: {words}", f"문장: {self.composer.text}"]
