import argparse
import math
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass

try:
    import cv2
    import mediapipe as mp
except ModuleNotFoundError as error:
    missing_package = error.name
    print(
        f"Missing dependency: {missing_package}. "
        "Install the required packages with: pip install opencv-python mediapipe"
    )
    sys.exit(1)


CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_NAME = "Face Mood Impression"
MAX_FACES = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
SMOOTHING_FRAMES = 8

MOOD_COLORS = {
    "Smile": (0, 220, 0),
    "Surprise": (0, 180, 255),
    "Angry": (0, 0, 255),
    "Sad": (255, 80, 80),
    "Neutral": (255, 255, 0),
}


@dataclass
class AppConfig:
    camera_index: int = CAMERA_INDEX
    camera_width: int = CAMERA_WIDTH
    camera_height: int = CAMERA_HEIGHT
    window_width: int = WINDOW_WIDTH
    window_height: int = WINDOW_HEIGHT
    max_faces: int = MAX_FACES
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE
    smoothing_frames: int = SMOOTHING_FRAMES
    show_mesh: bool = True
    show_debug: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time face mood impression detector.")
    parser.add_argument("--camera", type=int,
                        default=CAMERA_INDEX, help="Webcam index to open.")
    parser.add_argument("--width", type=int,
                        default=CAMERA_WIDTH, help="Requested camera width.")
    parser.add_argument("--height", type=int,
                        default=CAMERA_HEIGHT, help="Requested camera height.")
    parser.add_argument("--window-width", type=int,
                        default=WINDOW_WIDTH, help="Display window width.")
    parser.add_argument("--window-height", type=int,
                        default=WINDOW_HEIGHT, help="Display window height.")
    parser.add_argument("--max-faces", type=int, default=MAX_FACES,
                        help="Maximum number of faces to track.")
    parser.add_argument("--smoothing", type=int, default=SMOOTHING_FRAMES,
                        help="Frames used to smooth mood labels.")
    parser.add_argument("--debug", action="store_true",
                        help="Show calculated face ratios.")
    parser.add_argument("--no-mesh", action="store_true",
                        help="Hide face contour mesh.")
    args = parser.parse_args()

    return AppConfig(
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        window_width=args.window_width,
        window_height=args.window_height,
        max_faces=max(1, args.max_faces),
        smoothing_frames=max(1, args.smoothing),
        show_debug=args.debug,
        show_mesh=not args.no_mesh,
    )


def landmark_point(landmarks, index, width, height):
    point = landmarks.landmark[index]
    return int(point.x * width), int(point.y * height)


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def get_face_box(landmarks, width, height):
    xs = [point.x for point in landmarks.landmark]
    ys = [point.y for point in landmarks.landmark]

    x1 = max(0, int(min(xs) * width))
    y1 = max(0, int(min(ys) * height))
    x2 = min(width - 1, int(max(xs) * width))
    y2 = min(height - 1, int(max(ys) * height))

    return x1, y1, x2 - x1, y2 - y1


def get_face_center(face_box):
    x, y, w, h = face_box
    return x + w // 2, y + h // 2


def detect_mood(landmarks, width, height):
    left_mouth = landmark_point(landmarks, 61, width, height)
    right_mouth = landmark_point(landmarks, 291, width, height)
    upper_lip = landmark_point(landmarks, 13, width, height)
    lower_lip = landmark_point(landmarks, 14, width, height)
    left_cheek = landmark_point(landmarks, 234, width, height)
    right_cheek = landmark_point(landmarks, 454, width, height)

    left_eye_top = landmark_point(landmarks, 159, width, height)
    right_eye_top = landmark_point(landmarks, 386, width, height)
    left_brow = landmark_point(landmarks, 105, width, height)
    right_brow = landmark_point(landmarks, 334, width, height)
    left_inner_brow = landmark_point(landmarks, 107, width, height)
    right_inner_brow = landmark_point(landmarks, 336, width, height)
    left_outer_brow = landmark_point(landmarks, 70, width, height)
    right_outer_brow = landmark_point(landmarks, 300, width, height)

    face_width = max(distance(left_cheek, right_cheek), 1)
    mouth_width = distance(left_mouth, right_mouth) / face_width
    mouth_open = distance(upper_lip, lower_lip) / face_width
    brow_gap = ((left_eye_top[1] - left_brow[1]) +
                (right_eye_top[1] - right_brow[1])) / (2 * face_width)

    mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
    mouth_corner_y = (left_mouth[1] + right_mouth[1]) / 2
    corner_drop = (mouth_corner_y - mouth_center_y) / face_width
    inner_brow_drop = (
        (left_inner_brow[1] - left_outer_brow[1])
        + (right_inner_brow[1] - right_outer_brow[1])
    ) / (2 * face_width)

    angry_score = 0
    if brow_gap < 0.070:
        angry_score += 1
    if brow_gap < 0.058:
        angry_score += 2
    if inner_brow_drop > 0.000:
        angry_score += 1
    if inner_brow_drop > 0.010:
        angry_score += 2
    if mouth_open < 0.080:
        angry_score += 1
    if mouth_width < 0.420:
        angry_score += 1
    if corner_drop < 0.020:
        angry_score += 1

    debug_values = {
        "mouth_width": mouth_width,
        "mouth_open": mouth_open,
        "brow_gap": brow_gap,
        "inner_brow_drop": inner_brow_drop,
        "corner_drop": corner_drop,
        "angry_score": angry_score,
    }

    if mouth_open > 0.085 and brow_gap > 0.075:
        return "Surprise", debug_values
    if mouth_width > 0.43 and mouth_open < 0.075 and corner_drop < 0.010:
        return "Smile", debug_values
    if angry_score >= 5:
        return "Angry", debug_values
    if corner_drop > 0.018 and mouth_width < 0.39:
        return "Sad", debug_values
    return "Neutral", debug_values


def draw_face_impression(frame, x, y, mood, color):
    center = (x + 38, max(44, y - 52))
    cv2.circle(frame, center, 28, color, 2)
    cv2.circle(frame, (center[0] - 10, center[1] - 8), 3, color, -1)
    cv2.circle(frame, (center[0] + 10, center[1] - 8), 3, color, -1)

    if mood == "Smile":
        cv2.ellipse(frame, (center[0], center[1] + 4),
                    (14, 10), 0, 10, 170, color, 2)
    elif mood == "Surprise":
        cv2.circle(frame, (center[0], center[1] + 8), 7, color, 2)
    elif mood == "Angry":
        cv2.line(frame, (center[0] - 16, center[1] - 16),
                 (center[0] - 4, center[1] - 10), color, 2)
        cv2.line(frame, (center[0] + 4, center[1] - 10),
                 (center[0] + 16, center[1] - 16), color, 2)
        cv2.line(frame, (center[0] - 12, center[1] + 12),
                 (center[0] + 12, center[1] + 8), color, 2)
    elif mood == "Sad":
        cv2.ellipse(frame, (center[0], center[1] + 18),
                    (13, 10), 0, 200, 340, color, 2)
    else:
        cv2.line(frame, (center[0] - 12, center[1] + 10),
                 (center[0] + 12, center[1] + 10), color, 2)


def draw_label(frame, text, x, y, color):
    y = max(30, y)
    label_width = max(180, cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0][0] + 22)
    cv2.rectangle(frame, (x, y - 34), (x + label_width, y + 8), color, -1)
    cv2.putText(frame, text, (x + 10, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2)


def draw_panel(frame, lines, x=18, y=18):
    if not lines:
        return

    line_height = 26
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[
                0][0] for line in lines) + 28
    height = line_height * len(lines) + 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 14, y + 27 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )


def draw_debug_values(frame, values):
    lines = [
        f"Angry score: {values['angry_score']}",
        f"Brow gap: {values['brow_gap']:.3f}",
        f"Inner brow: {values['inner_brow_drop']:.3f}",
        f"Corner drop: {values['corner_drop']:.3f}",
        f"Mouth width: {values['mouth_width']:.3f}",
        f"Mouth open: {values['mouth_open']:.3f}",
    ]
    draw_panel(frame, lines, x=18, y=78)


def most_common_mood(history, fallback):
    if not history:
        return fallback
    return Counter(history).most_common(1)[0][0]


def get_history_for_face(face_histories, face_center, max_distance, smoothing_frames):
    best_key = None
    best_distance = max_distance

    for key in face_histories:
        center_distance = distance(face_center, key)
        if center_distance < best_distance:
            best_key = key
            best_distance = center_distance

    if best_key is None:
        return face_center, deque(maxlen=smoothing_frames)

    return best_key, face_histories.pop(best_key)


def prepare_window(config):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.window_width, config.window_height)


def resize_for_display(frame, config):
    frame_height, frame_width = frame.shape[:2]
    scale = min(config.window_width / frame_width,
                config.window_height / frame_height)
    display_width = int(frame_width * scale)
    display_height = int(frame_height * scale)

    if display_width == frame_width and display_height == frame_height:
        return frame
    return cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)


def open_camera(config):
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    return cap


def main():
    config = parse_args()
    cap = open_camera(config)
    if cap is None:
        print(f"Could not open webcam at index {config.camera_index}.")
        return

    prepare_window(config)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(
        thickness=1, circle_radius=1, color=(80, 255, 80))
    previous_time = time.time()
    smoothed_fps = 0
    face_histories = {}
    fullscreen = False
    show_mesh = config.show_mesh
    show_debug = config.show_debug

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=config.max_faces,
            refine_landmarks=True,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as face_mesh:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True

                next_face_histories = {}
                first_debug_values = None

                if results.multi_face_landmarks:
                    max_tracking_distance = width * 0.18

                    for face_landmarks in results.multi_face_landmarks:
                        detected_mood, debug_values = detect_mood(
                            face_landmarks, width, height)
                        face_box = get_face_box(face_landmarks, width, height)
                        face_center = get_face_center(face_box)
                        _, mood_history = get_history_for_face(
                            face_histories,
                            face_center,
                            max_tracking_distance,
                            config.smoothing_frames,
                        )
                        mood_history.append(detected_mood)
                        mood = most_common_mood(mood_history, detected_mood)
                        next_face_histories[face_center] = mood_history

                        color = MOOD_COLORS[mood]
                        x, y, w, h = face_box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        draw_label(frame, f"Mood: {mood}", x, y - 8, color)
                        draw_face_impression(frame, x, y, mood, color)

                        if first_debug_values is None:
                            first_debug_values = debug_values

                        if show_mesh:
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=drawing_spec,
                                connection_drawing_spec=drawing_spec,
                            )

                face_histories = next_face_histories

                current_time = time.time()
                instant_fps = 1 / \
                    (current_time - previous_time) if current_time != previous_time else 0
                smoothed_fps = instant_fps if smoothed_fps == 0 else (
                    smoothed_fps * 0.9) + (instant_fps * 0.1)
                previous_time = current_time

                draw_panel(
                    frame,
                    [
                        f"FPS: {smoothed_fps:.0f}",
                        f"Faces: {len(face_histories)}",
                        "Q/ESC quit | F fullscreen | M mesh | D debug",
                    ],
                )

                if show_debug and first_debug_values:
                    draw_debug_values(frame, first_debug_values)

                cv2.imshow(WINDOW_NAME, resize_for_display(frame, config))
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord("f"):
                    fullscreen = not fullscreen
                    window_mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, window_mode)
                    if not fullscreen:
                        cv2.resizeWindow(
                            WINDOW_NAME, config.window_width, config.window_height)
                if key == ord("m"):
                    show_mesh = not show_mesh
                if key == ord("d"):
                    show_debug = not show_debug
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
