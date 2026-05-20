import cv2
import mediapipe as mp
import time

# Initialize MediaPipe modules
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup drawing styles
face_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
hand_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0,0,255))

# Initialize webcam
cap = cv2.VideoCapture(0)

# FPS Counter
p_time = 0

# Initialize models
with mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue

        # Flip and convert BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face and hands
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_drawing_spec,
                    connection_drawing_spec=face_drawing_spec)

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_drawing_spec,
                    connection_drawing_spec=hand_drawing_spec)

        # Calculate FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Show FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Hand & Face Tracking', frame)

        # Break with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
