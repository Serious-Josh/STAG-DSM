# Hardware script to capture frames from video input then send to dedicated hardware running local_server.py

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import cv2
import requests


WINDOW_NAME = "DSM Client"
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi DSM streaming client")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--camera-path", type=str, default=None, help="Explicit camera path/device string")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://192.168.0.10:8000/infer",
        help="Inference server URL",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Max requests per second (defaults to 5.0 if --fps not provided)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Alias for --max-fps to match run_dsm heritage (takes precedence if provided)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display video with overlay information"
    )
    return parser.parse_args()


def resize_for_upload(frame):
    height, width = frame.shape[:2]
    target_width = 640
    if width <= target_width:
        return frame
    scale = target_width / float(width)
    target_height = int(height * scale)
    resized = cv2.resize(frame, (target_width, target_height))
    return resized


def encode_frame(frame) -> Optional[bytes]:
    success, buffer = cv2.imencode(".jpg", frame, JPEG_PARAMS)
    if not success:
        return None
    return buffer.tobytes()


def _backend_candidates():
    # On Windows, try several backends; on Linux (Pi), just use default.
    if os.name == "nt":
        return [
            (None, "DEFAULT"),
            (cv2.CAP_DSHOW, "CAP_DSHOW"),
            (cv2.CAP_MSMF, "CAP_MSMF"),
            (cv2.CAP_ANY, "CAP_ANY"),
        ]


def open_camera(index: int, device_path: Optional[str] = None):
    candidates = _backend_candidates()
    targets = []
    if device_path:
        targets.append((device_path, "path"))
    targets.append((index, "index"))
    for target_value, target_desc in targets:
        for backend_flag, backend_name in candidates:
            try:
                if backend_flag is None:
                    cap = cv2.VideoCapture(target_value)
                else:
                    cap = cv2.VideoCapture(target_value, backend_flag)
            except Exception:
                continue
            if cap is not None and cap.isOpened():
                print(f"[pi_run] Opened camera {target_desc} via backend {backend_name or 'DEFAULT'}.")
                return cap
            if cap is not None:
                cap.release()
    return None


def draw_overlay(
    frame,
    top_class: Optional[str],
    top_prob: float,
    probs: Optional[Dict[str, float]],
) -> None:
    display_text = "Waiting for prediction..."
    if top_class:
        display_text = f"{top_class} ({top_prob * 100:.1f}%)"
    cv2.putText(
        frame,
        display_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if not probs:
        return

    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    bar_height = 18
    gap = 8
    max_bars = min(len(sorted_probs), 6)
    frame_height, frame_width = frame.shape[:2]
    start_y = frame_height - (bar_height + gap) * max_bars - 10
    start_y = max(start_y, 40)

    for idx in range(max_bars):
        name, prob = sorted_probs[idx]
        bar_length = int((frame_width - 20) * max(prob, 0.0))
        y = start_y + idx * (bar_height + gap)
        cv2.rectangle(frame, (10, y), (10 + bar_length, y + bar_height), (255, 0, 0), -1)
        cv2.putText(
            frame,
            f"{name}: {prob * 100:.1f}%",
            (12, y + bar_height - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def send_frame(server_url: str, jpeg_bytes: bytes) -> Optional[Dict[str, Any]]:
    files = {"image": ("frame.jpg", jpeg_bytes, "image/jpeg")}
    try:
        response = requests.post(server_url, files=files, timeout=1.5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[WARN] Request failed: {exc}")
        return None


def main() -> None:
    args = parse_args()

    cap = open_camera(args.camera, device_path=args.camera_path)
    if cap is None or (not cap.isOpened()):
        print(f"[ERROR] Unable to open camera (index {args.camera}, path={args.camera_path})")
        sys.exit(1)

    max_fps = args.fps if args.fps is not None else args.max_fps
    if max_fps is None:
        max_fps = 5.0
    interval = 1.0 / max_fps if max_fps > 0 else 0.0
    last_send = 0.0
    last_top: Optional[str] = None
    last_prob = 0.0
    last_probs: Dict[str, float] = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera")
                time.sleep(0.05)
                continue

            current_time = time.time()
            should_send = interval == 0.0 or (current_time - last_send) >= interval

            if should_send:
                upload_frame = resize_for_upload(frame)
                jpeg_bytes = encode_frame(upload_frame)
                if jpeg_bytes is None:
                    print("[WARN] Failed to encode frame as JPEG")
                else:
                    result = send_frame(args.server_url, jpeg_bytes)
                    last_send = current_time
                    if result and "top_class" in result and "top_prob" in result:
                        last_top = str(result.get("top_class"))
                        top_prob_value: Any = result.get("top_prob", 0.0)
                        if isinstance(top_prob_value, (int, float)):
                            last_prob = float(top_prob_value)
                        else:
                            try:
                                last_prob = float(top_prob_value)
                            except (TypeError, ValueError):
                                last_prob = 0.0
                        probs = result.get("probs")
                        if isinstance(probs, dict):
                            cleaned = {}
                            for key, value in probs.items():
                                try:
                                    cleaned[str(key)] = float(value)
                                except (TypeError, ValueError):
                                    continue
                            last_probs = cleaned
                    elif result:
                        print("[WARN] Invalid response structure from server")

            display_frame = frame.copy()
            draw_overlay(display_frame, last_top, last_prob, last_probs)

            if args.show:
                cv2.imshow(WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
