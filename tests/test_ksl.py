import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from core.ksl import (ClipSegmenter, SentenceComposer, SequenceModel, encode, hand_frame)
from core.types import Hand, Landmark, Observation
from modes.sign_language import SignLanguageMode


def clip(kind=0, shift=0):
    frames = np.zeros((12, 2, 21, 3), dtype=np.float32)
    for t in range(12):
        for joint in range(21):
            frames[t, 0, joint] = [.2 + joint * .006 + shift,
                                    .3 + joint * .004 + kind * (joint % 5) * .025,
                                    joint * .001]
        frames[t, 0, :, 0] += t * .003
    return frames


class KSLTests(unittest.TestCase):
    def model(self):
        return SequenceModel.fit([
            {"label": label, "frames": clip(kind, shift)}
            for label, kind in (("WATER", 0), ("REQUEST", 1))
            for shift in (0, .003, -.003)])

    def test_model_roundtrip_and_rejection(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            model.save(path)
            restored = SequenceModel.load(path)
            self.assertEqual(restored.predict(clip(0))["label"], "WATER")
            self.assertEqual(restored.predict(clip(1))["label"], "REQUEST")
            self.assertIsNone(restored.predict(clip(9))["label"])

    def test_ambiguous_classes_rejected(self):
        model = self.model()
        model.templates[:] = encode(clip(0))
        self.assertIsNone(model.predict(clip(0))["label"])

    def test_fixed_handedness_slots(self):
        hand = Hand(tuple(Landmark(.2, .3, 0) for _ in range(21)), handedness="Right")
        frame = hand_frame(Observation((hand,)))
        self.assertFalse(frame[0].any())
        self.assertTrue(frame[1].any())
        self.assertFalse(hand_frame(Observation((hand, hand))).any())

    def test_segment_once_and_timeout(self):
        gate = ClipSegmenter()
        for t, frame in enumerate(clip()):
            self.assertIsNone(gate.update(frame, t * .05))
        empty = np.zeros((2, 21, 3))
        self.assertIsNotNone(gate.update(empty, 1))
        self.assertIsNone(gate.update(empty, 2))
        gate.update(clip()[0], 3)
        gate.update(clip()[0], 9)
        self.assertIsNone(gate.update(empty, 10))

    def test_sentence_and_full_mode(self):
        rules = {"rules": [{"tokens": ["WATER", "REQUEST"], "text": "물을 주세요."}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.model().save(root / "model.npz")
            (root / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
            mode = SignLanguageMode(root / "model.npz", root / "rules.json")
            now = 0
            for kind in (0, 1):
                for frame in clip(kind):
                    points = tuple(Landmark(*p) for p in frame[0])
                    mode.update(Observation((Hand(points, handedness="Left"),)), now)
                    now += .05
                now += .4
                mode.update(Observation(), now)
                now += .1
            mode.handle_key(13)
            self.assertEqual(mode.composer.text, "물을 주세요.")
            mode.handle_key(8)
            mode.handle_key(13)
            self.assertIn("지원하지", mode.composer.text)
            mode.handle_key(ord("c"))
            self.assertEqual(mode.composer.tokens, [])
            missing = SignLanguageMode(root / "missing.npz", root / "rules.json")
            missing.update(Observation())
            self.assertEqual(missing.composer.tokens, [])

    def test_bad_sequences(self):
        for frames in ([], np.zeros((5, 2, 21, 3)), np.full((5, 2, 21, 3), np.nan)):
            with self.assertRaises(ValueError):
                encode(frames)

    def test_training_cli_holds_out_signer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for signer in ("train", "heldout"):
                for label, kind in (("WATER", 0), ("REQUEST", 1)):
                    for i in range(3):
                        sample = {"version": 1, "mirrored": True, "signer": signer,
                                  "label": label, "frames": clip(kind, i * .001).tolist()}
                        (root / f"{signer}_{label}_{i}.json").write_text(json.dumps(sample), encoding="utf-8")
            output = root / "weights.npz"
            result = subprocess.run([sys.executable, "-B", "-m", "training.train_words",
                                     "--data", str(root), "--validation-signer", "heldout",
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["train_clips"], 6)
            self.assertEqual(report["validation_clips"], 6)
            self.assertEqual(len(SequenceModel.load(output).labels), 6)


if __name__ == "__main__":
    unittest.main()
