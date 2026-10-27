"""Virtual pointer with motion smoothing and debounced pinch clicks."""
from math import exp, hypot

import pyautogui
from core.geometry import normalize_to_roi, pinch_ratio
from core.swipe import OpenPalmSwipe
from modes.app_launcher import launch_app


class PCController:
    def __init__(self, *, backend=None, pinch_close=0.45, pinch_open=0.55,
                 drag_threshold=15, switch_timeout=2.5, launcher=None):
        if not 0 < pinch_close < pinch_open:
            raise ValueError("Expected 0 < pinch_close < pinch_open")
        if not 0 < drag_threshold < float("inf"):
            raise ValueError("Expected a finite drag_threshold > 0")
        if not 0 < switch_timeout < float("inf"):
            raise ValueError("Expected a finite switch_timeout > 0")
        self._mouse = backend if backend is not None else pyautogui
        self._mouse.FAILSAFE = True
        self._screen_w, self._screen_h = self._mouse.size()
        self.pinch_close = pinch_close
        self.pinch_open = pinch_open
        self.drag_threshold = drag_threshold
        self._mouse_down = False
        self.switch_timeout = switch_timeout
        self._alt_down = False
        self._switch_since = None
        self._swipe = OpenPalmSwipe()
        self._launcher = launcher if launcher is not None else launch_app
        self._last_launch = float("-inf")
        self.reset()

    def reset(self):
        try:
            self._release_button()
        finally:
            self._finish_switch()
        self._launch_since = None
        self._launch_gesture = None
        self._launch_latched = False
        self._launch_release_since = None
        self._swipe.reset()
        self._position = None
        self._filtered = None
        self._last_update = None
        self._pinch_since = None
        self._open_since = None
        self._pinched = False
        self._ready = False
        self._fist_since = None
        self._fist_last = None
        self._fist_hand = None
        self._zoom_gesture = None
        self._zoom_since = None
        self._zoom_last_seen = None
        self._zoom_last_step = None
        self._zoom_hand = None

    def _update_zoom(self, hand, now_s):
        gesture = (hand.gesture if hand and len(hand.landmarks) >= 21
                   and hand.score >= 0.7 else None)
        if gesture not in ("Thumb_Up", "Thumb_Down"):
            if self._zoom_gesture is not None:
                self.reset()
                return "Zoom ended: open thumb + middle to arm pointer"
            return None
        if (gesture != self._zoom_gesture or self._zoom_last_seen is None
                or not 0 <= now_s - self._zoom_last_seen <= 0.2
                or hand.handedness != self._zoom_hand):
            # Do not leave a mouse button held while issuing zoom shortcuts.
            self.reset()
            self._zoom_gesture = gesture
            self._zoom_since = now_s
            self._zoom_hand = hand.handedness
        self._zoom_last_seen = now_s
        direction = "in" if gesture == "Thumb_Up" else "out"
        if (now_s - self._zoom_since >= 0.5
                and (self._zoom_last_step is None
                     or now_s - self._zoom_last_step >= 0.6)):
            self._mouse.hotkey("win", "add" if direction == "in" else "subtract")
            self._zoom_last_step = now_s
            return f"Screen zoom {direction}"
        return f"Hold thumb {'up' if direction == 'in' else 'down'} to zoom {direction}"

    def _finish_switch(self):
        if not self._alt_down:
            return
        failsafe = self._mouse.FAILSAFE
        try:
            self._mouse.FAILSAFE = False
            self._mouse.keyUp("alt")
            self._alt_down = False
            self._switch_since = None
        finally:
            self._mouse.FAILSAFE = failsafe

    def _release_button(self):
        self._dragging = False
        self._pinch_origin = None
        self._cursor_origin = None
        if not self._mouse_down:
            return
        # Cleanup must also work when a screen corner triggered the fail-safe.
        failsafe = self._mouse.FAILSAFE
        try:
            self._mouse.FAILSAFE = False
            self._mouse.mouseUp(button="left")
            self._mouse_down = False
        finally:
            self._mouse.FAILSAFE = failsafe

    def update(self, hand, now_s):
        if not self._alt_down:
            zoom_status = self._update_zoom(hand, now_s)
            if zoom_status is not None:
                return zoom_status
        if (self._alt_down and hand and len(hand.landmarks) >= 21
                and hand.gesture == "Closed_Fist" and hand.score >= 0.7):
            if (self._fist_since is None or self._fist_last is None
                    or not 0 <= now_s - self._fist_last <= 0.2
                    or self._fist_hand != hand.handedness):
                self._fist_since = now_s
            self._fist_last = now_s
            self._fist_hand = hand.handedness
            if now_s - self._fist_since >= 0.3:
                self.reset()
                return "Window selected: fist confirmed"
        else:
            self._fist_since = None
            self._fist_last = None
            self._fist_hand = None
        if (self._alt_down and self._switch_since is not None
                and now_s - self._switch_since >= self.switch_timeout):
            self.reset()
            return "Window selected"
        shortcuts = {"Victory": ("chatgpt", "ChatGPT", "V"),
                     "ILoveYou": ("paint", "Paint", "I-love-you")}
        shortcut = (shortcuts.get(hand.gesture)
                    if hand is not None and hand.score >= 0.80 else None)
        if shortcut:
            self._release_button()
            self._finish_switch()
            app_key, app_name, gesture_name = shortcut
            self._launch_release_since = None
            if self._launch_since is None or self._launch_gesture != hand.gesture:
                self._launch_since = now_s
            self._launch_gesture = hand.gesture
            # Reserve this gesture for launching, with no pointer or pinch input.
            self._swipe.reset()
            self._position = self._filtered = self._last_update = None
            self._pinch_since = self._open_since = None
            self._pinched = self._ready = False
            if self._launch_latched:
                return "Release launch gesture to rearm"
            elapsed = now_s - self._launch_since
            if elapsed >= 1.0 and now_s - self._last_launch >= 5.0:
                self._launch_latched = True
                self._last_launch = now_s
                try:
                    opened = self._launcher(app_key)
                except (OSError, RuntimeError):
                    return f"Could not open {app_name}; check app installation"
                return (f"{app_name} opened" if opened else
                        f"Could not open {app_name}; check app installation")
            return f"Hold {gesture_name} 1s: {app_name} | {min(elapsed, 1.0):.1f}s"
        self._launch_since = None
        self._launch_gesture = None
        if self._launch_release_since is None:
            self._launch_release_since = now_s
        if now_s - self._launch_release_since >= 0.3:
            self._launch_latched = False
        switched = self._swipe.update(hand, now_s)
        if self._swipe.active or self._alt_down:
            self._release_button()
            # Keep pointer and pinch state out of an app-switch gesture.
            self._position = None
            self._filtered = None
            self._last_update = None
            self._pinch_since = None
            self._open_since = None
            self._pinched = False
            self._ready = False
            if switched:
                try:
                    if not self._alt_down:
                        self._alt_down = True
                        self._mouse.keyDown("alt")
                    self._mouse.press("tab")
                    self._switch_since = now_s
                except Exception:
                    self._finish_switch()
                    raise
            if self._alt_down:
                return ("Select window: swipe again; hold fist 0.3s "
                        f"or pause {self.switch_timeout:g}s to confirm")
            return "Open palm: swipe right to left to switch app"
        ratio = pinch_ratio(hand, finger_tip=12)
        if ratio is None:
            latch, release = self._launch_latched, self._launch_release_since
            self.reset()
            self._launch_latched, self._launch_release_since = latch, release
            return "No hand / landmarks"

        dt = (1 / 30 if self._last_update is None
              else max(0.0, min(now_s - self._last_update, 0.1)))
        self._last_update = now_s

        index = hand.landmarks[8]
        x, y = normalize_to_roi(index.x, index.y)
        target = (5 + x * (self._screen_w - 10),
                  5 + y * (self._screen_h - 10))

        # Require a deliberate open hand before arming, including after reset.
        if ratio >= self.pinch_open:
            self._release_button()
            self._pinch_since = None
            if self._open_since is None:
                self._open_since = now_s
            if now_s - self._open_since >= 0.10:
                self._pinched = False
                self._ready = True
        else:
            self._open_since = None

        if ratio <= self.pinch_close:
            if self._pinch_since is None:
                self._pinch_since = now_s
                # Freeze from the first closing frame, before debounce completes.
                if not self._mouse_down:
                    self._pinch_origin = target
                    self._cursor_origin = self._position
            if (self._ready and not self._pinched
                    and now_s - self._pinch_since >= 0.08):
                self._pinched = True
                self._ready = False
                # Track ownership before sending input so cleanup can release
                # even if the backend raises partway through the operation.
                self._mouse_down = True
                self._mouse.mouseDown(button="left")
        else:
            self._pinch_since = None

        allow_move = ratio > self.pinch_close and not self._mouse_down
        if self._mouse_down and self._pinch_origin is not None:
            dx = target[0] - self._pinch_origin[0]
            dy = target[1] - self._pinch_origin[1]
            if hypot(dx, dy) >= self.drag_threshold:
                self._dragging = True
            allow_move = self._dragging
            if self._cursor_origin is not None:
                # Relative movement avoids a jump when the pinch pose shifts.
                target = (max(5, min(self._screen_w - 5, self._cursor_origin[0] + dx)),
                          max(5, min(self._screen_h - 5, self._cursor_origin[1] + dy)))

        if allow_move:
            if self._filtered is None:
                self._filtered = target
            else:
                fx, fy = self._filtered
                # Slow small movements; follow intentional large moves faster.
                distance = hypot(target[0] - fx, target[1] - fy)
                tau = 0.12 if distance < 80 else 0.045
                alpha = 1 - exp(-dt / tau)
                self._filtered = (fx + alpha * (target[0] - fx),
                                  fy + alpha * (target[1] - fy))
            if (self._position is None
                    or hypot(self._filtered[0] - self._position[0],
                             self._filtered[1] - self._position[1]) >= 4):
                self._position = self._filtered
                self._mouse.moveTo(*self._position, duration=0)

        state = "READY" if self._ready else "Open thumb + middle to arm"
        if self._mouse_down:
            state = ("DRAG" if self._dragging else "CLICK held")
            state += ": separate thumb + middle to release"
        return f"{state} | T+M {ratio:.2f} (close {self.pinch_close:.2f})"
