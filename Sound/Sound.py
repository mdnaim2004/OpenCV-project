import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame

# =========================
# CONFIG
# =========================
AUDIO_FILE = "11.mp3"     # put correct path if not same folder
CAM_INDEX = 0
WINDOW_NAME = "Hand Volume Control (Device-Independent)"

# Hand distance -> volume mapping (default; you can calibrate with 'c')
MIN_DIST_DEFAULT = 20
MAX_DIST_DEFAULT = 200

# Smoothing (0.05 - 0.35 usually good)
ALPHA = 0.18

# Jitter deadzone (ignore tiny changes)
DEADZONE = 0.015   # volume change threshold (0.0-1.0)

# Volume clamp
VOL_MIN = 0.0
VOL_MAX = 1.0

# MediaPipe config
MAX_NUM_HANDS = 1
MODEL_COMPLEXITY = 0  # 0 = faster, 1 = more accurate

# =========================
# INIT AUDIO (pygame)
# =========================
pygame.mixer.init()
pygame.mixer.music.load(AUDIO_FILE)
pygame.mixer.music.play(-1)
pygame.mixer.music.set_volume(0.5)

is_paused = False
is_muted = False
last_volume = 0.5

# =========================
# INIT MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# INIT CAMERA
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Camera not found! Try changing CAM_INDEX or check permissions.")

# Trackbar fallback
cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar("Manual Volume", WINDOW_NAME, 50, 100, lambda x: None)

# =========================
# STATE
# =========================
min_dist = MIN_DIST_DEFAULT
max_dist = MAX_DIST_DEFAULT

smooth_vol = 0.5
prev_set_vol = smooth_vol

# FPS
prev_time = time.time()
fps = 0.0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def set_app_volume(vol_0_1: float):
    """Set pygame music volume (device-independent)."""
    global last_volume, prev_set_vol, is_muted

    vol_0_1 = float(clamp(vol_0_1, VOL_MIN, VOL_MAX))

    # deadzone to prevent jitter updates
    if abs(vol_0_1 - prev_set_vol) < DEADZONE:
        return

    if not is_muted:
        pygame.mixer.music.set_volume(vol_0_1)
        last_volume = vol_0_1

    prev_set_vol = vol_0_1

def toggle_pause():
    global is_paused
    if is_paused:
        pygame.mixer.music.unpause()
        is_paused = False
    else:
        pygame.mixer.music.pause()
        is_paused = True

def toggle_mute():
    global is_muted
    if is_muted:
        # restore
        pygame.mixer.music.set_volume(last_volume)
        is_muted = False
    else:
        pygame.mixer.music.set_volume(0.0)
        is_muted = True

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Calibration helper
def calibrate_from_current(length):
    """
    Set a reasonable range around current pinch distance.
    You can press 'c' multiple times at different distances for better calibration.
    """
    global min_dist, max_dist
    # Expand range gently
    min_dist = int(min(min_dist, length * 0.7))
    max_dist = int(max(max_dist, length * 1.3))
    # Safety clamp
    min_dist = int(clamp(min_dist, 5, 300))
    max_dist = int(clamp(max_dist, min_dist + 20, 500))

print("Controls: q=quit | p=play/pause | m=mute/unmute | c=calibrate range")

while True:
    ok, img = cap.read()
    if not ok:
        continue

    img = cv2.flip(img, 1)  # mirror
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_found = False
    vol_percent = None
    current_length = None

    if results.multi_hand_landmarks:
        hand_found = True
        for hand_landmark in results.multi_hand_landmarks:
            # Landmark list (pixel coords)
            lm = hand_landmark.landmark
            thumb = (int(lm[4].x * w), int(lm[4].y * h))   # thumb tip
            index = (int(lm[8].x * w), int(lm[8].y * h))   # index tip

            current_length = distance(thumb, index)

            # Map distance -> vol (0..1)
            target_vol = float(np.interp(current_length, [min_dist, max_dist], [0.0, 1.0]))
            target_vol = clamp(target_vol, 0.0, 1.0)

            # Smooth volume
            smooth_vol = (1 - ALPHA) * smooth_vol + ALPHA * target_vol
            set_app_volume(smooth_vol)

            vol_percent = int(smooth_vol * 100)

            # Draw UI elements
            cv2.circle(img, thumb, 10, (255, 0, 0), -1)
            cv2.circle(img, index, 10, (255, 0, 0), -1)
            cv2.line(img, thumb, index, (0, 255, 0), 3)
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

            break  # max 1 hand

    else:
        # Manual fallback (trackbar)
        manual = cv2.getTrackbarPos("Manual Volume", WINDOW_NAME) / 100.0
        # only apply manual if hand not found
        smooth_vol = (1 - ALPHA) * smooth_vol + ALPHA * manual
        set_app_volume(smooth_vol)
        vol_percent = int(smooth_vol * 100)

    # Volume bar
    bar_y = int(np.interp(smooth_vol, [0.0, 1.0], [400, 150]))
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
    cv2.rectangle(img, (50, bar_y), (85, 400), (0, 255, 0), -1)

    # FPS calc
    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    # Status text
    status = []
    status.append(f"APP Volume: {vol_percent}%")
    status.append("Hand: ON" if hand_found else "Hand: OFF (Manual)")
    status.append(f"Muted: {is_muted} | Paused: {is_paused}")
    status.append(f"Range: {min_dist}-{max_dist}")
    status.append(f"FPS: {fps:.1f}")

    y = 35
    for s in status:
        cv2.putText(img, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        y += 26

    if current_length is not None:
        cv2.putText(img, f"Dist: {current_length:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        toggle_pause()
    elif key == ord('m'):
        toggle_mute()
    elif key == ord('c'):
        # Calibrate using current pinch distance if available
        if current_length is not None:
            calibrate_from_current(current_length)

cap.release()
cv2.destroyAllWindows()
hands.close()