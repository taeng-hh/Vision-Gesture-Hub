# Architecture

```text
webcam / OpenCV
      |
      v
core.detector.GestureDetector  (MediaPipe -> neutral dataclasses)
      |
      v
core.mode_manager.ModeManager (exactly one mode)
      |              |                 |
      v              v                 v
modes.sign_language  modes.pc_control  modes.presentation
   (placeholder)       |                 |
                       v                 v
               OS mouse actions    OS keyboard actions
```

**Design rules**
- `IDLE` is the home screen. Point the index fingertip at a mode card for 0.8 seconds to enter.
- `core/navigation.py` handles dwell selection in normalized camera coordinates; `ui/overlay.py` draws cards, progress, and an internal pointer.
- In a mode, dwell on `HOME` to return. PC and presentation modes also have an `ENABLE` / `DISABLE` control. Leave the control before selecting again.
- Navigation controls take priority over mode actions. Entering a mode keeps OS actions off until explicitly enabled.
- `core/` must not launch programs or send input events.
- `modes/sign_language.py` must never execute OS actions.
- `modes/pc_control.py` and `modes/presentation.py` are gated by `actions_enabled`.
- Switching modes disables OS actions until the operator explicitly enables them again.
- Existing MediaPipe hand-shape labels are **not** Korean Sign Language translations.
- Landmarks are normalized image coordinates; physical screen touch requires separate calibration and depth sensing.
- For future continuous sign recognition, collect consented sequences of both hands plus pose/face features with signer-independent validation; single-frame CSV alone is insufficient.
