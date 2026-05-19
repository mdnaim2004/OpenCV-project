"""
Air Writer - OpenCV + MediaPipe

Gesture guide
-------------
WRITE       Index finger up only
ERASE       Index + middle + ring fingers up
CLEAR       Open palm, held for about half a second
IDLE        Fist or unsupported gesture

Controls
--------
Q / ESC     Quit
C           Clear canvas
S           Save canvas as PNG
U           Undo last stroke / erase / clear
F           Toggle fullscreen
1-6         Change pen color
+ / -       Change brush size
[ / ]       Change eraser size
H           Toggle hand landmarks
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    width: int = 960
    height: int = 540
    fps: int = 60
    draw_smoothing: int = 3
    min_detection: float = 0.6
    min_tracking: float = 0.6
    model_complexity: int = 0
    process_scale: float = 0.5
    fast_move_threshold: int = 45
    clear_hold_seconds: float = 0.55
    undo_limit: int = 12
    output_dir: Path = Path("captures")
    fullscreen: bool = True


@dataclass
class AppState:
    canvas: np.ndarray
    color_idx: int = 0
    brush_size: int = 8
    eraser_size: int = 50
    previous_gesture: str = "IDLE"
    previous_draw_point: tuple[int, int] | None = None
    points: Deque[tuple[int, int]] = field(default_factory=deque)
    undo_stack: Deque[np.ndarray] = field(default_factory=deque)
    clear_started_at: float | None = None
    clear_done_for_gesture: bool = False
    previous_erase_point: tuple[int, int] | None = None
    save_count: int = 0
    show_landmarks: bool = False
    fullscreen: bool = True


PEN_COLORS = [
    (255, 255, 255),  # white
    (0, 255, 255),    # yellow
    (57, 255, 20),    # green
    (255, 50, 50),    # blue
    (255, 0, 255),    # magenta
    (0, 165, 255),    # orange
]
COLOR_NAMES = ["White", "Yellow", "Green", "Blue", "Magenta", "Orange"]

GESTURE_COLORS = {
    "WRITE": (57, 255, 20),
    "ERASE": (0, 100, 255),
    "CLEAR": (0, 50, 200),
    "IDLE": (100, 100, 100),
    "UNKNOWN": (80, 80, 80),
}

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_JOINTS = [3, 6, 10, 14, 18]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
WINDOW_NAME = "Air Writer"


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Draw in the air with OpenCV and MediaPipe.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open.")
    parser.add_argument("--width", type=int, default=960, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=540, help="Requested camera height.")
    parser.add_argument("--fps", type=int, default=60, help="Requested camera FPS.")
    parser.add_argument("--smoothing", type=int, default=3, help="Number of recent points used for slow-motion smoothing.")
    parser.add_argument("--min-detection", type=float, default=0.6, help="MediaPipe detection confidence.")
    parser.add_argument("--min-tracking", type=float, default=0.6, help="MediaPipe tracking confidence.")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=0, help="MediaPipe model complexity. 0 is faster, 1 is more accurate.")
    parser.add_argument("--process-scale", type=float, default=0.5, help="Hand-tracking frame scale. Lower is faster, higher is more accurate.")
    parser.add_argument("--clear-hold", type=float, default=0.55, help="Seconds an open palm must be held before clearing.")
    parser.add_argument("--output-dir", type=Path, default=Path("captures"), help="Folder for saved PNG files.")
    parser.add_argument("--windowed", action="store_true", help="Start in a normal resizable window instead of fullscreen.")
    args = parser.parse_args()
    return Settings(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        draw_smoothing=max(1, args.smoothing),
        min_detection=args.min_detection,
        min_tracking=args.min_tracking,
        model_complexity=args.model_complexity,
        process_scale=float(np.clip(args.process_scale, 0.35, 1.0)),
        clear_hold_seconds=max(0.2, args.clear_hold),
        output_dir=args.output_dir,
        fullscreen=not args.windowed,
    )


def open_camera(settings: Settings) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {settings.camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def get_fingers_up(landmarks, handedness: str = "Right") -> list[bool]:
    fingers = []

    if handedness == "Right":
        fingers.append(landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_JOINTS[0]].x)
    else:
        fingers.append(landmarks[FINGER_TIPS[0]].x > landmarks[FINGER_JOINTS[0]].x)

    for tip, joint in zip(FINGER_TIPS[1:], FINGER_JOINTS[1:]):
        fingers.append(landmarks[tip].y < landmarks[joint].y)

    return fingers


def get_gesture(fingers: list[bool]) -> str:
    _, index, middle, ring, pinky = fingers

    if not any(fingers):
        return "IDLE"
    if all(fingers):
        return "CLEAR"
    if index and not middle and not ring and not pinky:
        return "WRITE"
    if index and middle and ring and not pinky:
        return "ERASE"
    return "UNKNOWN"


def prepare_tracking_frame(frame: np.ndarray, process_scale: float) -> np.ndarray:
    if process_scale >= 0.99:
        tracking_frame = frame
    else:
        h, w = frame.shape[:2]
        size = (max(1, int(w * process_scale)), max(1, int(h * process_scale)))
        tracking_frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return rgb


def push_undo(state: AppState, settings: Settings) -> None:
    state.undo_stack.append(state.canvas.copy())
    while len(state.undo_stack) > settings.undo_limit:
        state.undo_stack.popleft()


def undo(state: AppState) -> bool:
    if not state.undo_stack:
        return False
    state.canvas[:] = state.undo_stack.pop()
    state.previous_draw_point = None
    state.previous_erase_point = None
    state.points.clear()
    return True


def smooth_point(points: Deque[tuple[int, int]]) -> tuple[int, int]:
    weights = np.linspace(1.0, 2.0, num=len(points))
    x_values = np.array([p[0] for p in points])
    y_values = np.array([p[1] for p in points])
    return int(np.average(x_values, weights=weights)), int(np.average(y_values, weights=weights))


def adaptive_point(
    points: Deque[tuple[int, int]],
    previous_point: tuple[int, int] | None,
    fast_move_threshold: int,
) -> tuple[int, int]:
    raw_x, raw_y = points[-1]
    if previous_point is None or len(points) < 2:
        return raw_x, raw_y

    distance = float(np.hypot(raw_x - previous_point[0], raw_y - previous_point[1]))
    smoothed_x, smoothed_y = smooth_point(points)

    if distance >= fast_move_threshold:
        raw_weight = min(0.88, 0.55 + (distance - fast_move_threshold) / 220)
        return (
            int(raw_x * raw_weight + smoothed_x * (1.0 - raw_weight)),
            int(raw_y * raw_weight + smoothed_y * (1.0 - raw_weight)),
        )

    return smoothed_x, smoothed_y


def composite_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 5, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.where(mask_3ch > 0, canvas, frame)


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (220, 220, 220),
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_color_palette(frame: np.ndarray, state: AppState) -> None:
    x = 190
    for idx, color in enumerate(PEN_COLORS):
        top_left = (x + idx * 34, 15)
        bottom_right = (x + idx * 34 + 24, 45)
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        border = (255, 255, 255) if idx == state.color_idx else (80, 80, 80)
        cv2.rectangle(frame, top_left, bottom_right, border, 2)

    draw_text(frame, COLOR_NAMES[state.color_idx], (405, 36), 0.55)


def draw_ui(frame: np.ndarray, state: AppState, gesture: str, fps: int) -> np.ndarray:
    h, w = frame.shape[:2]
    frame = composite_canvas(frame, state.canvas)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 68), (15, 15, 15), -1)
    cv2.rectangle(overlay, (0, h - 34), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.76, frame, 0.24, 0)

    mode_color = GESTURE_COLORS.get(gesture, (150, 150, 150))
    label = gesture if gesture != "UNKNOWN" else "READY"
    cv2.rectangle(frame, (10, 10), (170, 55), mode_color, -1)
    cv2.rectangle(frame, (10, 10), (170, 55), (255, 255, 255), 1)
    cv2.putText(frame, label, (28, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    draw_color_palette(frame, state)

    pen_color = PEN_COLORS[state.color_idx]
    draw_text(frame, f"Pen {state.brush_size}", (540, 36))
    cv2.circle(frame, (625, 31), max(2, state.brush_size // 2), pen_color, -1)
    draw_text(frame, f"Eraser {state.eraser_size}", (665, 36))
    draw_text(frame, f"Undo {len(state.undo_stack)}", (785, 36))
    draw_text(frame, f"FPS {fps:02d}", (w - 95, 38), 0.6, (180, 255, 150))

    hints = "[Q] Quit  [C] Clear  [S] Save  [U] Undo  [F] Fullscreen  [1-6] Color  [+/-] Pen  [[/]] Eraser"
    draw_text(frame, hints, (10, h - 12), 0.43, (165, 165, 165))
    return frame


def draw_cursor(frame: np.ndarray, cx: int, cy: int, gesture: str, color: tuple[int, int, int], brush_size: int, eraser_size: int) -> None:
    if gesture == "WRITE":
        cv2.circle(frame, (cx, cy), brush_size // 2 + 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(2, brush_size // 2), color, -1, cv2.LINE_AA)
    elif gesture == "ERASE":
        cv2.circle(frame, (cx, cy), eraser_size, (0, 0, 220), 2, cv2.LINE_AA)
        draw_text(frame, "ERASE", (cx - 25, cy + 5), 0.42, (0, 0, 255))


def draw_stroke_segment(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    brush_size: int,
) -> None:
    distance = int(np.hypot(end[0] - start[0], end[1] - start[1]))
    steps = max(1, distance // max(brush_size // 2, 1))
    for step in range(1, steps + 1):
        alpha = step / steps
        x = int(start[0] + (end[0] - start[0]) * alpha)
        y = int(start[1] + (end[1] - start[1]) * alpha)
        cv2.circle(canvas, (x, y), max(1, brush_size // 2), color, -1, cv2.LINE_AA)
    cv2.line(canvas, start, end, color, brush_size, cv2.LINE_AA)


def erase_segment(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    eraser_size: int,
) -> None:
    distance = int(np.hypot(end[0] - start[0], end[1] - start[1]))
    step_size = max(eraser_size // 2, 1)
    steps = max(1, distance // step_size)
    for step in range(steps + 1):
        alpha = step / steps
        x = int(start[0] + (end[0] - start[0]) * alpha)
        y = int(start[1] + (end[1] - start[1]) * alpha)
        cv2.circle(canvas, (x, y), eraser_size, (0, 0, 0), -1, cv2.LINE_AA)


def save_canvas(state: AppState, settings: Settings) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    state.save_count += 1
    path = settings.output_dir / f"air_canvas_{timestamp}_{state.save_count:03d}.png"
    cv2.imwrite(str(path), state.canvas)
    return path


def apply_window_mode(state: AppState, frame_width: int, frame_height: int) -> None:
    if state.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(frame_width, 1280), min(frame_height, 720))


def handle_key(key: int, state: AppState, settings: Settings) -> bool:
    if key in (ord("q"), 27):
        print("Exiting...")
        return False
    if key == ord("c"):
        push_undo(state, settings)
        state.canvas[:] = 0
        print("Canvas cleared.")
    elif key == ord("s"):
        path = save_canvas(state, settings)
        print(f"Saved: {path}")
    elif key == ord("u"):
        print("Undo." if undo(state) else "Nothing to undo.")
    elif key == ord("f"):
        state.fullscreen = not state.fullscreen
        apply_window_mode(state, state.canvas.shape[1], state.canvas.shape[0])
        print("Fullscreen on." if state.fullscreen else "Windowed mode.")
    elif ord("1") <= key <= ord(str(len(PEN_COLORS))):
        state.color_idx = key - ord("1")
        print(f"Color: {COLOR_NAMES[state.color_idx]}")
    elif key in (ord("+"), ord("=")):
        state.brush_size = min(48, state.brush_size + 2)
    elif key == ord("-"):
        state.brush_size = max(2, state.brush_size - 2)
    elif key == ord("["):
        state.eraser_size = max(12, state.eraser_size - 5)
    elif key == ord("]"):
        state.eraser_size = min(140, state.eraser_size + 5)
    elif key == ord("h"):
        state.show_landmarks = not state.show_landmarks
    return True


def handle_gesture(state: AppState, settings: Settings, gesture: str, cx: int, cy: int) -> None:
    if gesture == "WRITE":
        if state.previous_gesture != "WRITE":
            push_undo(state, settings)
            state.previous_erase_point = None
        state.points.append((cx, cy))
        while len(state.points) > settings.draw_smoothing:
            state.points.popleft()

        point = adaptive_point(state.points, state.previous_draw_point, settings.fast_move_threshold)
        if state.previous_draw_point is not None:
            draw_stroke_segment(state.canvas, state.previous_draw_point, point, PEN_COLORS[state.color_idx], state.brush_size)
        state.previous_draw_point = point
    else:
        state.previous_draw_point = None
        state.points.clear()

    if gesture == "ERASE":
        if state.previous_gesture != "ERASE":
            push_undo(state, settings)
            state.previous_erase_point = (cx, cy)
        erase_segment(state.canvas, state.previous_erase_point or (cx, cy), (cx, cy), state.eraser_size)
        state.previous_erase_point = (cx, cy)
    else:
        state.previous_erase_point = None

    if gesture == "CLEAR":
        now = time.perf_counter()
        if state.clear_started_at is None:
            state.clear_started_at = now
        elif not state.clear_done_for_gesture and now - state.clear_started_at >= settings.clear_hold_seconds:
            push_undo(state, settings)
            state.canvas[:] = 0
            state.clear_done_for_gesture = True
            print("Canvas cleared.")
    else:
        state.clear_started_at = None
        state.clear_done_for_gesture = False


def main() -> None:
    cv2.setUseOptimized(True)

    settings = parse_args()
    cap = open_camera(settings)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or settings.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or settings.height

    state = AppState(
        canvas=np.zeros((actual_h, actual_w, 3), dtype=np.uint8),
        points=deque(maxlen=settings.draw_smoothing),
        undo_stack=deque(maxlen=settings.undo_limit),
        fullscreen=settings.fullscreen,
    )
    fps_counter: Deque[float] = deque(maxlen=20)
    fps = 0

    print(__doc__)
    print(
        f"Camera started at {actual_w}x{actual_h}, target {settings.fps} FPS, "
        f"tracking scale {settings.process_scale:.2f}. Show your hand to begin."
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    apply_window_mode(state, actual_w, actual_h)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=settings.model_complexity,
        min_detection_confidence=settings.min_detection,
        min_tracking_confidence=settings.min_tracking,
    )

    try:
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            tracking_rgb = prepare_tracking_frame(frame, settings.process_scale)
            result = hands.process(tracking_rgb)

            gesture = "IDLE"
            cx, cy = -1, -1

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_lm = result.multi_hand_landmarks[0]
                handedness = result.multi_handedness[0].classification[0].label

                if state.show_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                landmarks = hand_lm.landmark
                gesture = get_gesture(get_fingers_up(landmarks, handedness))
                cx = int(np.clip(landmarks[8].x * w, 0, w - 1))
                cy = int(np.clip(landmarks[8].y * h, 0, h - 1))

                handle_gesture(state, settings, gesture, cx, cy)
                draw_cursor(frame, cx, cy, gesture, PEN_COLORS[state.color_idx], state.brush_size, state.eraser_size)
            else:
                state.previous_draw_point = None
                state.previous_erase_point = None
                state.points.clear()
                state.clear_started_at = None
                state.clear_done_for_gesture = False

            state.previous_gesture = gesture

            frame = draw_ui(frame, state, gesture, fps)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not handle_key(key, state, settings):
                break

            loop_seconds = time.perf_counter() - start
            fps_counter.append(loop_seconds)
            fps = int(round(1.0 / (np.mean(fps_counter) + 1e-9)))
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
