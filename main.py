import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import urllib.request
import shutil
import subprocess

def download_model_if_missing(model_path):
    if os.path.exists(model_path):
        return
    print(f"Downloading MediaPipe Face Landmarker model to {model_path}...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed via urllib: {e}")
        if os.path.exists(model_path):
            os.unlink(model_path)
        # Fall back to curl
        if shutil.which("curl") is not None:
            print("Retrying download using system curl...")
            try:
                subprocess.run(["curl", "-fL", "-o", model_path, url], check=True)
                print("Download complete via curl.")
                return
            except Exception as e2:
                print(f"Download failed via curl: {e2}")
                if os.path.exists(model_path):
                    os.unlink(model_path)
        raise RuntimeError(
            f"Could not download model file. Please download it manually from:\n"
            f"  {url}\n"
            f"and place it at:\n"
            f"  {os.path.abspath(model_path)}"
        )


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _smoothing_alpha(cutoff_hz, dt_s):
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt_s)

    def filter(self, x, t_s):
        if not math.isfinite(x):
            return float(self._x_prev) if self._x_prev is not None else float(x)

        if self._x_prev is None or self._t_prev is None:
            self._x_prev = float(x)
            self._t_prev = float(t_s)
            return float(x)

        dt = float(t_s) - self._t_prev
        if dt <= 0:
            return float(self._x_prev)

        dx = (x - self._x_prev) / dt
        alpha_d = self._smoothing_alpha(self._d_cutoff, dt)
        dx_smooth = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        cutoff = self._min_cutoff + self._beta * abs(dx_smooth)
        alpha = self._smoothing_alpha(cutoff, dt)

        x_smooth = alpha * x + (1.0 - alpha) * self._x_prev

        self._x_prev = float(x_smooth)
        self._dx_prev = float(dx_smooth)
        self._t_prev = float(t_s)

        return float(x_smooth)


class OneEuro2D:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self):
        self._fx.reset()
        self._fy.reset()

    def filter(self, x, y, t_s):
        return self._fx.filter(x, t_s), self._fy.filter(y, t_s)




def features_from_landmarks(landmarks, yaw, pitch, roll):
    # Left eye: outer is 263, inner is 362
    left_outer = (landmarks[263].x, landmarks[263].y)
    left_inner = (landmarks[362].x, landmarks[362].y)
    left_center = ((left_outer[0] + left_inner[0]) / 2.0, (left_outer[1] + left_inner[1]) / 2.0)
    
    # Right eye: outer is 33, inner is 133
    right_outer = (landmarks[33].x, landmarks[33].y)
    right_inner = (landmarks[133].x, landmarks[133].y)
    right_center = ((right_outer[0] + right_inner[0]) / 2.0, (right_outer[1] + right_inner[1]) / 2.0)
    
    # Left iris center is 473, right iris center is 468
    left_iris = (landmarks[473].x, landmarks[473].y)
    right_iris = (landmarks[468].x, landmarks[468].y)
    
    # Offsets in normalized coordinates
    left_iris_offset_x = left_iris[0] - left_center[0]
    left_iris_offset_y = left_iris[1] - left_center[1]
    right_iris_offset_x = right_iris[0] - right_center[0]
    right_iris_offset_y = right_iris[1] - right_center[1]
    
    return np.array([
        left_iris_offset_x,
        left_iris_offset_y,
        right_iris_offset_x,
        right_iris_offset_y,
        float(yaw),
        float(pitch),
        float(roll)
    ], dtype=np.float64)


def expand_features(X):
    # X can be shape (N, 7) or (7,)
    is_1d = (X.ndim == 1)
    if is_1d:
        X = np.expand_dims(X, axis=0)
    
    cols = [X]
    # Iris indices: 0, 1, 2, 3. Head indices: 4, 5, 6.
    for i in [0, 1, 2, 3]:
        for j in [4, 5, 6]:
            cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
            
    expanded = np.hstack(cols)
    if is_1d:
        return expanded[0]
    return expanded


def calibrate_gaze(calibration_data, lambda_val=0.001):
    # calibration_data is a list of tuples: (tx, ty, l_ox, l_oy, r_ox, r_oy, yaw, pitch, roll)
    if len(calibration_data) < 20:
        return None, None, None, None
        
    X_raw = []
    Y_x = []
    Y_y = []
    
    for item in calibration_data:
        tx, ty, l_ox, l_oy, r_ox, r_oy, yaw, pitch, roll = item
        X_raw.append([l_ox, l_oy, r_ox, r_oy, yaw, pitch, roll])
        Y_x.append(tx)
        Y_y.append(ty)
        
    X_raw = np.array(X_raw, dtype=np.float64)  # (N, 7)
    Y_x = np.array(Y_x, dtype=np.float64)
    Y_y = np.array(Y_y, dtype=np.float64)
    
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    # Threshold below 1e-9 to prevent divide-by-zero
    stds = np.where(stds < 1e-9, 1.0, stds)
    
    X_std = (X_raw - means) / stds
    X_expanded = expand_features(X_std)  # (N, 19)
    
    # Add bias column at the end (shape N, 20)
    X_aug = np.hstack([X_expanded, np.ones((X_expanded.shape[0], 1))])
    
    # Ridge: do not regularize the bias term (index -1)
    n_features_aug = X_aug.shape[1]
    alpha_mat = lambda_val * np.eye(n_features_aug)
    alpha_mat[-1, -1] = 0.0
    
    XtX = X_aug.T @ X_aug + alpha_mat
    
    try:
        w_x = np.linalg.solve(XtX, X_aug.T @ Y_x)
        w_y = np.linalg.solve(XtX, X_aug.T @ Y_y)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        w_x, _, _, _ = np.linalg.lstsq(XtX, X_aug.T @ Y_x, rcond=None)
        w_y, _, _, _ = np.linalg.lstsq(XtX, X_aug.T @ Y_y, rcond=None)
        
    return w_x, w_y, means, stds


def predict_gaze(feats, x_weights, y_weights, means, stds):
    # feats is a 7-D base features vector
    if x_weights is None or y_weights is None or means is None or stds is None:
        return None
        
    # Standardize base features
    feats_std = (feats - means) / stds
    
    # Expand to 19-D
    feats_expanded = expand_features(feats_std)
    
    # Add bias term (1.0) at the end to match w_x, w_y
    feats_aug = np.append(feats_expanded, 1.0)
    
    pred_x = np.dot(feats_aug, x_weights)
    pred_y = np.dot(feats_aug, y_weights)
    
    return int(pred_x), int(pred_y)

cap = cv2.VideoCapture(0)

# Configure native fullscreen window
window_name = "AffectIQ Eye-Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Dummy render to initialize and query monitor dimensions
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imshow(window_name, dummy)
cv2.waitKey(15)  # Wait for window manager to apply fullscreen

rect = cv2.getWindowImageRect(window_name)
if rect is not None and rect[2] > 0 and rect[3] > 0:
    screen_w, screen_h = rect[2], rect[3]
else:
    screen_w, screen_h = 1920, 1080  # Default fallback
print(f"Native Screen Resolution: {screen_w}x{screen_h}")

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

model_path = "face_landmarker_v2_with_blendshapes.task"
download_model_if_missing(model_path)

options = mp_vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)

# Calibration states
STATE_PREVIEW = 0
STATE_CALIB_STATIONARY = 1
STATE_CALIB_SHAKING = 2
STATE_CALIBRATED = 3

current_state = STATE_PREVIEW
active_point_idx = 0
calibration_data = []  # Stores: (target_x, target_y, left_x, left_y, right_x, right_y, yaw, pitch, roll)

# Ridge coefficients & standardization weights
x_weights = None
y_weights = None
means = None
stds = None

gaze_filter = OneEuro2D(min_cutoff=1.0, beta=0.05)
point_start_time = 0.0
point_running = False
point_captured_duration = 0.0
last_time = time.time()
latest_features = None  # Stores: np.array([left_x, left_y, right_x, right_y, yaw, pitch, roll])

# Compute 9 calibration points dynamically for screen size (10%, 50%, 90% scaling)
margin_x = int(screen_w * 0.1)
margin_y = int(screen_h * 0.1)
mid_x = screen_w // 2
mid_y = screen_h // 2

CALIBRATION_POINTS = [
    (margin_x, margin_y),             # Top-Left
    (mid_x, margin_y),                # Top-Center
    (screen_w - margin_x, margin_y),  # Top-Right
    (margin_x, mid_y),                # Middle-Left
    (mid_x, mid_y),                   # Center
    (screen_w - margin_x, mid_y),     # Middle-Right
    (margin_x, screen_h - margin_y),  # Bottom-Left
    (mid_x, screen_h - margin_y),     # Bottom-Center
    (screen_w - margin_x, screen_h - margin_y)  # Bottom-Right
]

while True:
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    success, frame = cap.read()
    if not success:
        break

    latest_features = None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            h, w, _ = frame.shape

            # Eyelids index: Left top lid: 159, Left bottom lid: 145. Corners: Outer 263, Inner 362.
            # Right top lid: 386, Right bottom lid: 374. Corners: Outer 33, Inner 133.
            left_eye_h = abs(face_landmarks[159].y - face_landmarks[145].y)
            left_eye_w = math.sqrt((face_landmarks[263].x - face_landmarks[362].x)**2 + (face_landmarks[263].y - face_landmarks[362].y)**2)
            left_ear = left_eye_h / left_eye_w if left_eye_w > 0 else 0.0

            right_eye_h = abs(face_landmarks[386].y - face_landmarks[374].y)
            right_eye_w = math.sqrt((face_landmarks[33].x - face_landmarks[133].x)**2 + (face_landmarks[33].y - face_landmarks[133].y)**2)
            right_ear = right_eye_h / right_eye_w if right_eye_w > 0 else 0.0

            # Only process features if eyes are open (blink gate)
            if left_ear >= 0.15 and right_ear >= 0.15:
                # Decompose rigid transform matrix into yaw, pitch, roll
                yaw, pitch, roll = 0.0, 0.0, 0.0
                if result.facial_transformation_matrixes:
                    mat = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
                    if mat.shape == (4, 4):
                        r = mat[:3, :3]
                        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))
                        yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
                        roll = math.degrees(math.atan2(r[1, 0], r[1, 1]))

                latest_features = features_from_landmarks(face_landmarks, yaw, pitch, roll)

            # Draw iris and eye corner overlays on the PiP frame
            h_f, w_f, _ = frame.shape
            left_center_px = (int(face_landmarks[473].x * w_f), int(face_landmarks[473].y * h_f))
            right_center_px = (int(face_landmarks[468].x * w_f), int(face_landmarks[468].y * h_f))
            cv2.circle(frame, left_center_px, 6, (255, 0, 0), -1)
            cv2.circle(frame, right_center_px, 6, (255, 0, 0), -1)

            # Draw outer/inner corners lines
            left_c1_px = (int(face_landmarks[362].x * w_f), int(face_landmarks[362].y * h_f))
            left_c2_px = (int(face_landmarks[263].x * w_f), int(face_landmarks[263].y * h_f))
            cv2.line(frame, left_c1_px, left_c2_px, (0, 255, 0), 1)

            right_c1_px = (int(face_landmarks[33].x * w_f), int(face_landmarks[33].y * h_f))
            right_c2_px = (int(face_landmarks[133].x * w_f), int(face_landmarks[133].y * h_f))
            cv2.line(frame, right_c1_px, right_c2_px, (0, 255, 0), 1)

            # Draw head pose degrees
            if 'yaw' in locals():
                cv2.putText(
                    frame,
                    f"Yaw:{yaw:+.1f} Pitch:{pitch:+.1f} Roll:{roll:+.1f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

    # Construct fullscreen visualization canvas
    canvas_w, canvas_h = screen_w, screen_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (30, 26, 26)  # Minimalist dark background

    if current_state == STATE_PREVIEW:
        # Render centered preview feed
        preview_w, preview_h = 640, 480
        resized_frame = cv2.resize(frame, (preview_w, preview_h))
        x_offset = (canvas_w - preview_w) // 2
        y_offset = (canvas_h - preview_h) // 2
        canvas[y_offset:y_offset+preview_h, x_offset:x_offset+preview_w] = resized_frame

        # Frame border
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset+preview_w, y_offset+preview_h), (0, 255, 0), 2)

        # Instructions
        cv2.putText(canvas, "Align your face in the box. Press 'C' to start calibration.", 
                    (canvas_w // 2 - 270, y_offset + preview_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

    elif current_state == STATE_CALIB_STATIONARY:
        is_tracking_active = (latest_features is not None)

        if not point_running:
            # Waiting for SPACE key
            target_color = (255, 255, 255)  # White
            instruction_text = "Stage 1/2 (Stationary): Look at target and press SPACE to start point..."
            text_color = target_color
            
            # Draw grid
            for idx, pt in enumerate(CALIBRATION_POINTS):
                if idx == active_point_idx:
                    cv2.circle(canvas, pt, 24, target_color, 2, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 8, target_color, -1, cv2.LINE_AA)
                    cv2.line(canvas, (pt[0]-35, pt[1]), (pt[0]+35, pt[1]), target_color, 1)
                    cv2.line(canvas, (pt[0], pt[1]-35), (pt[0], pt[1]+35), target_color, 1)
                else:
                    cv2.circle(canvas, pt, 6, (80, 80, 80), -1, cv2.LINE_AA)
        else:
            # Active point calibration
            if is_tracking_active:
                point_captured_duration += dt
                target_color = (0, 255, 0)  # Green
                instruction_text = "Stage 1/2 (Stationary): Keep head still and hold gaze..."
                text_color = target_color
            else:
                # Paused due to blink or face lost
                target_color = (0, 0, 255)  # Red
                instruction_text = "TRACKING PAUSED - PLEASE OPEN EYES AND LOOK AT TARGET"
                text_color = target_color

            # Check point completion
            if point_captured_duration >= 5.0:
                active_point_idx += 1
                point_running = False
                point_captured_duration = 0.0
                if active_point_idx >= len(CALIBRATION_POINTS):
                    # Transition to Stage 2 (Head Shaking)
                    current_state = STATE_CALIB_SHAKING
                    active_point_idx = 0
                    print("Stage 1 complete. Transitioning to Stage 2 (Head Shaking).")
            else:
                # Render 9-point layout
                for idx, pt in enumerate(CALIBRATION_POINTS):
                    if idx == active_point_idx:
                        cv2.circle(canvas, pt, 8, target_color, -1, cv2.LINE_AA)
                        cv2.line(canvas, (pt[0]-35, pt[1]), (pt[0]+35, pt[1]), target_color, 1)
                        cv2.line(canvas, (pt[0], pt[1]-35), (pt[0], pt[1]+35), target_color, 1)

                        sweep_ratio = min(point_captured_duration / 5.0, 1.0)
                        sweep_angle = int(sweep_ratio * 360)
                        cv2.ellipse(canvas, pt, (24, 24), 0, -90, -90 + sweep_angle, target_color, 2, cv2.LINE_AA)

                        # Continuously log data points after 1.5s settle-down delay
                        if is_tracking_active and point_captured_duration >= 1.5:
                            calibration_data.append((
                                pt[0], pt[1], 
                                latest_features[0], latest_features[1],
                                latest_features[2], latest_features[3],
                                latest_features[4], latest_features[5], latest_features[6]
                            ))
                    else:
                        # Faint uncalibrated points
                        cv2.circle(canvas, pt, 6, (80, 80, 80), -1, cv2.LINE_AA)

        # Guidance instruction text aligned to fullscreen height
        cv2.putText(canvas, f"Point {active_point_idx + 1} / 9", (40, canvas_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, instruction_text, (360, canvas_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

    elif current_state == STATE_CALIB_SHAKING:
        is_tracking_active = (latest_features is not None)

        if not point_running:
            # Waiting for SPACE key
            target_color = (255, 255, 255)  # White
            instruction_text = "Stage 2/2 (Head Shake): Look at target and press SPACE to start point..."
            text_color = target_color
            
            # Draw grid
            for idx, pt in enumerate(CALIBRATION_POINTS):
                if idx == active_point_idx:
                    cv2.circle(canvas, pt, 24, target_color, 2, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 8, target_color, -1, cv2.LINE_AA)
                    cv2.line(canvas, (pt[0]-35, pt[1]), (pt[0]+35, pt[1]), target_color, 1)
                    cv2.line(canvas, (pt[0], pt[1]-35), (pt[0], pt[1]+35), target_color, 1)
                else:
                    cv2.circle(canvas, pt, 6, (80, 80, 80), -1, cv2.LINE_AA)
        else:
            # Active point calibration
            if is_tracking_active:
                point_captured_duration += dt
                target_color = (0, 140, 255)  # BGR Orange
                instruction_text = "Stage 2/2 (Head Shake): Hold gaze & slowly move/shake head..."
                text_color = target_color
            else:
                # Paused due to blink or face lost
                target_color = (0, 0, 255)  # Red
                instruction_text = "TRACKING PAUSED - PLEASE OPEN EYES AND LOOK AT TARGET"
                text_color = target_color

            # Check point completion
            if point_captured_duration >= 5.0:
                active_point_idx += 1
                point_running = False
                point_captured_duration = 0.0
                if active_point_idx >= len(CALIBRATION_POINTS):
                    # Run the 19-parameter Ridge Regression solver
                    x_weights, y_weights, means, stds = calibrate_gaze(calibration_data)
                    current_state = STATE_CALIBRATED
                    gaze_filter.reset()
                    print(f"Calibration completed. Collected {len(calibration_data)} frames.")
            else:
                # Render 9-point layout
                for idx, pt in enumerate(CALIBRATION_POINTS):
                    if idx == active_point_idx:
                        cv2.circle(canvas, pt, 8, target_color, -1, cv2.LINE_AA)
                        cv2.line(canvas, (pt[0]-35, pt[1]), (pt[0]+35, pt[1]), target_color, 1)
                        cv2.line(canvas, (pt[0], pt[1]-35), (pt[0], pt[1]+35), target_color, 1)

                        sweep_ratio = min(point_captured_duration / 5.0, 1.0)
                        sweep_angle = int(sweep_ratio * 360)
                        cv2.ellipse(canvas, pt, (24, 24), 0, -90, -90 + sweep_angle, target_color, 2, cv2.LINE_AA)

                        # Continuously log data points after 1.5s settle-down delay
                        if is_tracking_active and point_captured_duration >= 1.5:
                            calibration_data.append((
                                pt[0], pt[1], 
                                latest_features[0], latest_features[1],
                                latest_features[2], latest_features[3],
                                latest_features[4], latest_features[5], latest_features[6]
                            ))
                    else:
                        # Faint uncalibrated points
                        cv2.circle(canvas, pt, 6, (80, 80, 80), -1, cv2.LINE_AA)

        # Guidance instruction text aligned to fullscreen height
        cv2.putText(canvas, f"Point {active_point_idx + 1} / 9", (40, canvas_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, instruction_text, (360, canvas_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

    elif current_state == STATE_CALIBRATED:
        # Draw 5 hollow Magenta validation rings (Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right)
        # to check calibration accuracy
        v_margin_x = int(canvas_w * 0.1)
        v_margin_y = int(canvas_h * 0.1)
        v_mid_x = canvas_w // 2
        v_mid_y = canvas_h // 2
        validation_targets = [
            ("Center", (v_mid_x, v_mid_y)),
            ("Top-Left", (v_margin_x, v_margin_y)),
            ("Top-Right", (canvas_w - v_margin_x, v_margin_y)),
            ("Bottom-Left", (v_margin_x, canvas_h - v_margin_y)),
            ("Bottom-Right", (canvas_w - v_margin_x, canvas_h - v_margin_y))
        ]

        for label, pt in validation_targets:
            cv2.circle(canvas, pt, 20, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, pt, 4, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, label, (pt[0] - 30, pt[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

        # Predict gaze coordinate in real-time
        if latest_features is not None:
            pred = predict_gaze(latest_features, x_weights, y_weights, means, stds)
            if pred is not None:
                pred_x, pred_y = pred
                pred_x = np.clip(pred_x, 0, canvas_w - 1)
                pred_y = np.clip(pred_y, 0, canvas_h - 1)

                # Smooth gaze predictions with OneEuroFilter to reduce jitter
                smooth_x, smooth_y = gaze_filter.filter(pred_x, pred_y, current_time)
                smooth_x = int(np.clip(smooth_x, 0, canvas_w - 1))
                smooth_y = int(np.clip(smooth_y, 0, canvas_h - 1))

                # Draw gaze cursor overlay
                cv2.circle(canvas, (smooth_x, smooth_y), 22, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (smooth_x, smooth_y), 6, (0, 140, 255), -1, cv2.LINE_AA)

                # Print coordinates
                cv2.putText(canvas, f"Gaze Coordinate: ({smooth_x}, {smooth_y})", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)

        # Status text aligned to fullscreen height
        cv2.putText(canvas, "Gaze Tracking Active. Press 'R' to recalibrate.", (40, canvas_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

    # PiP Camera Preview overlay during calibration states
    if current_state in (STATE_CALIB_STATIONARY, STATE_CALIB_SHAKING):
        pip_w, pip_h = 240, 180
        resized_frame = cv2.resize(frame, (pip_w, pip_h))
        y_start = 30
        y_end = 30 + pip_h
        x_start = canvas_w - 30 - pip_w
        x_end = canvas_w - 30
        canvas[y_start:y_end, x_start:x_end] = resized_frame
        cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), (200, 200, 200), 1)

    # Show window
    cv2.imshow(window_name, canvas)

    # Keyboard control handler
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        if (current_state == STATE_CALIB_STATIONARY or current_state == STATE_CALIB_SHAKING) and not point_running:
            point_running = True
            point_captured_duration = 0.0
            gaze_filter.reset()
            print(f"Starting Point {active_point_idx + 1} / 9 calibration.")
    elif key == ord('c') or key == ord('C'):
        if current_state == STATE_PREVIEW:
            current_state = STATE_CALIB_STATIONARY
            active_point_idx = 0
            calibration_data = []
            x_weights = None
            y_weights = None
            means = None
            stds = None
            point_running = False
            point_captured_duration = 0.0
            print("Calibration sequence initiated (Stage 1: Stationary). Press SPACE to begin first point.")
    elif key == ord('r') or key == ord('R'):
        current_state = STATE_PREVIEW
        active_point_idx = 0
        calibration_data = []
        x_weights = None
        y_weights = None
        means = None
        stds = None
        point_running = False
        point_captured_duration = 0.0
        gaze_filter.reset()
        print("Calibration reset.")


cap.release()
cv2.destroyAllWindows()