import math
import time

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
MAX_FACES = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
SHOW_DEBUG_VALUES = False

MOOD_COLORS = {
    "Smile": (0, 220, 0),
    "Surprise": (0, 180, 255),
    "Angry": (0, 0, 255),
    "Sad": (255, 80, 80),
    "Neutral": (255, 255, 0),
}


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
    brow_gap = ((left_eye_top[1] - left_brow[1]) + (right_eye_top[1] - right_brow[1])) / (2 * face_width)

    mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
    mouth_corner_y = (left_mouth[1] + right_mouth[1]) / 2
    corner_drop = (mouth_corner_y - mouth_center_y) / face_width
    inner_brow_drop = (
        (left_inner_brow[1] - left_outer_brow[1]) +
        (right_inner_brow[1] - right_outer_brow[1])
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
        cv2.ellipse(frame, (center[0], center[1] + 4), (14, 10), 0, 10, 170, color, 2)
    elif mood == "Surprise":
        cv2.circle(frame, (center[0], center[1] + 8), 7, color, 2)
    elif mood == "Angry":
        cv2.line(frame, (center[0] - 16, center[1] - 16), (center[0] - 4, center[1] - 10), color, 2)
        cv2.line(frame, (center[0] + 4, center[1] - 10), (center[0] + 16, center[1] - 16), color, 2)
        cv2.line(frame, (center[0] - 12, center[1] + 12), (center[0] + 12, center[1] + 8), color, 2)
    elif mood == "Sad":
        cv2.ellipse(frame, (center[0], center[1] + 18), (13, 10), 0, 200, 340, color, 2)
    else:
        cv2.line(frame, (center[0] - 12, center[1] + 10), (center[0] + 12, center[1] + 10), color, 2)


def draw_label(frame, text, x, y, color):
    y = max(30, y)
    cv2.rectangle(frame, (x, y - 28), (x + 190, y + 6), color, -1)
    cv2.putText(frame, text, (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)


def draw_debug_values(frame, values):
    lines = [
        f"Angry score: {values['angry_score']}",
        f"Brow gap: {values['brow_gap']:.3f}",
        f"Inner brow: {values['inner_brow_drop']:.3f}",
        f"Mouth width: {values['mouth_width']:.3f}",
        f"Mouth open: {values['mouth_open']:.3f}",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, 80 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(80, 255, 80))
    previous_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
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

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mood, debug_values = detect_mood(face_landmarks, width, height)
                    color = MOOD_COLORS[mood]
                    x, y, w, h = get_face_box(face_landmarks, width, height)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    draw_label(frame, f"Mood: {mood}", x, y - 8, color)
                    draw_face_impression(frame, x, y, mood, color)
                    if SHOW_DEBUG_VALUES:
                        draw_debug_values(frame, debug_values)

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )

            current_time = time.time()
            fps = 1 / (current_time - previous_time) if current_time != previous_time else 0
            previous_time = current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Face Mood Impression", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
