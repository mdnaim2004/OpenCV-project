"""
app.py
======
Hand Gesture Volume & Music Controller
─────────────────────────────────────────────────────────────────────
Two-hand gesture control for a pygame music player using MediaPipe.

LEFT HAND  (pinch)  → volume control
RIGHT HAND gestures →
  Fist         = Stop music
  Palm open    = Resume / play
  Thumbs up    = Like (console log)
  Swipe right  = Next track
  Swipe left   = Previous track

Keyboard shortcuts:
  q  – quit                 p – play/pause
  m  – mute/unmute          n – next track
  b  – previous track       s – settings panel
  t  – gesture tutorial     w  – toggle waveform
  l  – toggle landmarks     u  – toggle shuffle
  r  – toggle repeat        c  – calibrate pinch range
  d  – dataset collect mode  f  – flush dataset
  0-9 (in collect mode) – set gesture label
  + / - – increase/decrease smoothing alpha
  [ / ] – decrease/increase min/max dist

Usage:
  python app.py
  python app.py --music ./songs
  python app.py --cam 1 --theme light --no-waveform
"""

import sys
import os
import time
import argparse
import logging
import traceback

import cv2
import mediapipe as mp
import numpy as np

# ── local modules ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from modules.config_manager import load_config, save_config
from modules.logger_setup   import setup_logger
from modules.player         import MusicPlayer
from modules.gesture        import GestureDetector
from modules.ui_renderer    import UIRenderer
from modules.data_collector import DataCollector

WINDOW = "Hand Volume Control"


# ══════════════════════════════════════════════════════════════════ #
#  CLI argument parsing
# ══════════════════════════════════════════════════════════════════ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Hand Gesture Volume & Music Controller",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--music",       type=str,  default=None,
                   help="Path to music folder (default: config.json value)")
    p.add_argument("--cam",         type=int,  default=None,
                   help="Camera device index (default: 0)")
    p.add_argument("--theme",       type=str,  choices=["dark", "light"],
                   default=None, help="UI theme: dark | light")
    p.add_argument("--no-waveform", action="store_true",
                   help="Disable waveform animation")
    p.add_argument("--tutorial",    action="store_true",
                   help="Start with gesture tutorial visible")
    p.add_argument("--collect",     action="store_true",
                   help="Start in dataset-collection mode")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════ #
#  Camera initialisation with fallback
# ══════════════════════════════════════════════════════════════════ #

def open_camera(index: int, w: int, h: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        # Try a few common indices
        for idx in [0, 1, 2]:
            if idx == index:
                continue
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                logging.getLogger("HandVolumeControl").warning(
                    f"Camera {index} not found. Using camera {idx}.")
                break
        else:
            raise RuntimeError("No camera found. Check --cam argument or permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


# ══════════════════════════════════════════════════════════════════ #
#  Main
# ══════════════════════════════════════════════════════════════════ #

def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────
    cfg = load_config()

    # CLI overrides
    if args.music:
        cfg["audio"]["music_folder"] = args.music
    if args.cam is not None:
        cfg["camera"]["index"] = args.cam
    if args.theme:
        cfg["ui"]["theme"] = args.theme
    if args.no_waveform:
        cfg["ui"]["show_waveform"] = False
    if args.tutorial:
        cfg["ui"]["show_tutorial"] = True

    # ── Logger ────────────────────────────────────────────────────
    log_cfg = cfg.get("logging", {})
    logger  = setup_logger(log_cfg.get("level", "INFO"),
                           log_cfg.get("log_file", "app.log"))
    logger.info("Hand Volume Control starting…")

    # ── Sub-systems ───────────────────────────────────────────────
    player    = MusicPlayer(
        music_folder   = cfg["audio"]["music_folder"],
        initial_volume = cfg["audio"]["initial_volume"],
        shuffle        = cfg["audio"]["shuffle"],
        repeat         = cfg["audio"]["repeat"],
    )
    detector  = GestureDetector(cfg)
    renderer  = UIRenderer(cfg)
    collector = DataCollector()

    if args.collect:
        collector.toggle()

    cam_cfg = cfg["camera"]
    cap = open_camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"])

    # ── MediaPipe ─────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        max_num_hands          = 2,
        model_complexity       = 0,
        min_detection_confidence = 0.60,
        min_tracking_confidence  = 0.60,
    )

    # ── Window ────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("Manual Vol", WINDOW, 50, 100, lambda _: None)

    # ── State ─────────────────────────────────────────────────────
    frame_idx   = 0
    skip_n      = max(1, cam_cfg.get("process_every_n_frames", 2))
    fps         = 0.0
    prev_t      = time.time()
    last_result = None   # last GestureResult
    

    # Manual fallback volume smoothing
    manual_smooth = player.volume

    logger.info("Running. Controls: q=quit | p=play/pause | m=mute | "
                "n=next | b=prev | t=tutorial | s=settings | d=collect")
    print("\n  q=quit  p=pause  m=mute  n=next  b=prev  "
          "t=tutorial  s=settings  d=collect  c=calibrate\n")

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                logger.warning("Frame read failed – attempting reconnect…")
                cap.release()
                time.sleep(0.5)
                try:
                    cap = open_camera(cam_cfg["index"],
                                      cam_cfg["width"], cam_cfg["height"])
                except RuntimeError:
                    break
                continue

            img = cv2.flip(img, 1)
            h, w = img.shape[:2]


            # ── Process gesture every N frames ────────────────────
            frame_idx += 1
            if frame_idx % skip_n == 0:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks and results.multi_handedness:
                    last_result = detector.process(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                        w, h
                    )


                    # ── Apply volume (left hand) ───────────────────
                    if last_result.left_volume >= 0:
                        player.set_volume(last_result.left_volume)

                    # ── Act on right-hand gestures ─────────────────
                    if last_result.gesture_new:
                        gn = last_result.gesture_name
                        if gn == "fist":
                            player.stop()
                            renderer.notify("✊ Stopped")
                        elif gn == "palm":
                            if player.is_paused:
                                player.toggle_pause()
                                renderer.notify("▶ Resumed")
                            elif not results.multi_hand_landmarks:
                                player.toggle_pause()
                        elif gn == "thumbs_up":
                            info = player.get_current_info()
                            logger.info(f"👍 Liked: {info.get('title', '?')}")
                            renderer.notify(f"👍 Liked: {info.get('title', '?')[:24]}")
                        elif gn == "swipe_right":
                            player.next_track()
                            renderer.notify("→ Next Track")
                        elif gn == "swipe_left":
                            player.prev_track()
                            renderer.notify("← Prev Track")

                    # ── Dataset collection ─────────────────────────
                    if collector.active:
                        for hl in results.multi_hand_landmarks:
                            collector.record(hl)

                    # ── Draw landmarks ─────────────────────────────
                    if renderer.show_landmarks:
                        for hl in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                img, hl, mp_hands.HAND_CONNECTIONS,
                                mp_draw.DrawingSpec((0, 200, 255), 2, 2),
                                mp_draw.DrawingSpec((255, 100, 0),  2, 1),
                            )
                        # Pinch line on left hand (if detected)
                        if last_result and last_result.left_volume >= 0:
                            for hl, hd in zip(results.multi_hand_landmarks,
                                               results.multi_handedness):
                                if hd.classification[0].label == "Right":
                                    lm = hl.landmark
                                    t_pt = (int(lm[4].x*w), int(lm[4].y*h))
                                    i_pt = (int(lm[8].x*w), int(lm[8].y*h))
                                    cv2.line(img, t_pt, i_pt, (0, 255, 80), 3)
                                    cv2.circle(img, t_pt, 8, (0, 255, 80), -1)
                                    cv2.circle(img, i_pt, 8, (0, 255, 80), -1)
                else:
                    # No hands → manual trackbar fallback
                    manual = cv2.getTrackbarPos("Manual Vol", WINDOW) / 100.0
                    manual_smooth = 0.85 * manual_smooth + 0.15 * manual
                    player.set_volume(manual_smooth)
                    last_result = None


            # ── FPS ───────────────────────────────────────────────
            now = time.time()
            dt  = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt


            # ── Player update (track-end events) ──────────────────
            player.update()
            elapsed, _ = player.get_progress()
            info        = player.get_current_info()


            # ── Build render state ────────────────────────────────___
            state = {
                "volume":          player.volume,
                "is_muted":        player.is_muted,
                "is_paused":       player.is_paused,
                "shuffle":         player.shuffle,
                "repeat":          player.repeat,
                "hand_found_left": last_result is not None and last_result.left_volume >= 0,
                "hand_found_right":last_result is not None and last_result.gesture_name != "none",
                "fps":             fps,
                "track_info":      info,
                "elapsed_s":       elapsed,
                "gesture_name":    last_result.gesture_name if last_result else "none",
                "gesture_new":     last_result.gesture_new  if last_result else False,
                "min_dist":        detector.min_dist,
                "max_dist":        detector.max_dist,
                "alpha":           detector.alpha,
                "cfg":             cfg,
            }

            # Dataset overlay
            if collector.active:
                cv2.putText(img, collector.get_status(),
                            (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 220, 255), 1, cv2.LINE_AA)

            renderer.draw(img, state)
            cv2.imshow(WINDOW, img)

            # ── Keyboard ──────────────────────────────────────────__
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'):
                break
            elif key == ord('p'):
                player.toggle_pause()
                renderer.notify("⏸ Paused" if player.is_paused else "▶ Playing")
            elif key == ord('m'):
                player.toggle_mute()
                renderer.notify("🔇 Muted" if player.is_muted else "🔊 Unmuted")
            elif key == ord('n'):
                player.next_track()
                renderer.notify("→ Next Track")
            elif key == ord('b'):
                player.prev_track()
                renderer.notify("← Prev Track")
            elif key == ord('u'):
                player.toggle_shuffle()
                renderer.notify(f"⇌ Shuffle {'ON' if player.shuffle else 'OFF'}")
            elif key == ord('r'):
                player.toggle_repeat()
                renderer.notify(f"↻ Repeat {'ON' if player.repeat else 'OFF'}")
            elif key == ord('t'):
                renderer.toggle_tutorial()
            elif key == ord('s'):
                renderer.toggle_settings()
            elif key == ord('w'):
                renderer.show_waveform = not renderer.show_waveform
            elif key == ord('l'):
                renderer.show_landmarks = not renderer.show_landmarks
            elif key == ord('c'):
                # Calibrate using current pinch distance 
                if last_result and last_result.left_dist > 0:
                    detector.calibrate(last_result.left_dist)
                    renderer.notify(f"Calibrated: {detector.min_dist}–{detector.max_dist}")
            elif key == ord('+') or key == ord('='):
                detector.alpha = round(min(0.5, detector.alpha + 0.02), 3)
                renderer.notify(f"Alpha: {detector.alpha:.3f}")
            elif key == ord('-'):
                detector.alpha = round(max(0.02, detector.alpha - 0.02), 3)
                renderer.notify(f"Alpha: {detector.alpha:.3f}")
            elif key == ord('['):
                detector.min_dist = max(5, detector.min_dist - 5)
                renderer.notify(f"Min dist: {detector.min_dist}")
            elif key == ord(']'):
                detector.max_dist = min(500, detector.max_dist + 5)
                renderer.notify(f"Max dist: {detector.max_dist}")
            elif key == ord('<') or key == ord(','):
                detector.max_dist = max(detector.min_dist + 20, detector.max_dist - 5)
                renderer.notify(f"Max dist: {detector.max_dist}")
            elif key == ord('d'):
                on = collector.toggle()
                renderer.notify(f"Dataset collect: {'ON' if on else 'OFF'}")
            elif key == ord('f'):
                fp = collector.flush()
                renderer.notify(f"Saved dataset: {os.path.basename(fp) if fp else 'Nothing'}")
            elif collector.active and chr(key) in "0123456789":
                name = collector.set_label(chr(key))
                renderer.notify(f"Label → {chr(key)} ({name})")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        # ── Save config on exit ────────────────────────────────────
        cfg["gesture"]["min_dist"]        = detector.min_dist
        cfg["gesture"]["max_dist"]        = detector.max_dist
        cfg["gesture"]["smoothing_alpha"] = detector.alpha
        cfg["audio"]["shuffle"]           = player.shuffle
        cfg["audio"]["repeat"]            = player.repeat
        cfg["ui"]["show_waveform"]        = renderer.show_waveform
        cfg["ui"]["show_landmarks"]       = renderer.show_landmarks
        save_config(cfg)
        logger.info("Config saved.")

        # ── Flush any collected data ──────────────────────────────
        if collector.active and collector._rows:
            collector.flush()

        # ── Cleanup ──────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        import pygame
        pygame.mixer.quit()
        logger.info("Goodbye!")

if __name__ == "__main__":
    main()