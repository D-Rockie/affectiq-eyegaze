import cv2
import mediapipe as mp
import numpy as np
import time


class FeatureSmoother:
    def __init__(self, size=5):
        self.size = size
        self.history = []

    def add_and_smooth(self, feats):
        # feats is a tuple: (lh, lv, rh, rv, tx, ty, dist, yaw, pitch)
        self.history.append(feats)
        if len(self.history) > self.size:
            self.history.pop(0)
        arr = np.array(self.history)
        return tuple(np.mean(arr, axis=0))

    def reset(self):
        self.history = []




def get_gaze_direction(ratio):

    if ratio < 0.35:
        return "RIGHT"

    elif ratio > 0.65:
        return "LEFT"

    else:
        return "CENTER"

def get_vertical_direction(vertical_ratio):

    if vertical_ratio < 0.35:
        return "UP"

    elif vertical_ratio > 0.65:
        return "DOWN"

    else:
        return "CENTER"


def get_iris_center(face_landmarks, iris_indices, w, h):

    points = []

    for idx in iris_indices:

        landmark = face_landmarks.landmark[idx]

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        points.append((x, y))

    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)

    return int(center_x), int(center_y)

def calculate_iris_ratio(iris_x, c1_x, c2_x, eye_width):
    if eye_width == 0:
        return 0
    return (iris_x - min(c1_x, c2_x)) / eye_width

def calculate_vertical_ratio(iris_y, c1_y, c2_y, eye_width):
    if eye_width == 0:
        return 0
    baseline_y = (c1_y + c2_y) / 2.0
    return (iris_y - baseline_y) / eye_width


def extract_gaze_features(lh, lv, rh, rv, yaw, pitch, dist):
    """
    Extracts the 19-dimensional feature vector:
    - 4 Iris features
    - 3 Head pose features
    - 12 Cross-term interaction terms (iris * head)
    """
    iris_feats = [lh, lv, rh, rv]
    head_feats = [yaw, pitch, dist]
    cross_feats = []
    for i in iris_feats:
        for h in head_feats:
            cross_feats.append(i * h)
    return np.array(iris_feats + head_feats + cross_feats, dtype=float)


def calibrate_gaze(calibration_data, lambda_val=0.01):
    """
    Fits Ridge Regression mapping weights using standardized 19-D features.
    Uses direct augmented-matrix solving for numerical stability and a variance noise-gate.
    """
    if len(calibration_data) < 20:
        return None, None, None, None

    X_raw = []
    Y_x = []
    Y_y = []

    for item in calibration_data:
        tx, ty, lh, lv, rh, rv, h_tx, h_ty, h_dist, h_yaw, h_pitch = item
        X_raw.append(extract_gaze_features(lh, lv, rh, rv, h_yaw, h_pitch, h_dist))
        Y_x.append(tx)
        Y_y.append(ty)

    X_raw = np.array(X_raw)  # Shape (N, 19)
    Y_x = np.array(Y_x)      # Shape (N,)
    Y_y = np.array(Y_y)      # Shape (N,)

    # Feature standardization with a noise-gate threshold
    means = np.mean(X_raw, axis=0)
    stds = np.std(X_raw, axis=0)
    # Threshold at 0.01: do not scale features with very low variance (avoids noise amplification)
    stds[stds < 0.01] = 1.0  

    X_std = (X_raw - means) / stds

    # Add bias column (all ones)
    X_std_bias = np.hstack([np.ones((X_std.shape[0], 1)), X_std])  # Shape (N, 20)

    # Use highly stable Augmented Matrix method for Ridge Regression:
    # We append sqrt(lambda) * I to X, and zeros to Y
    # The bias column (index 0) has a regularization weight of 0.0
    reg_weights = np.ones(20) * np.sqrt(lambda_val)
    reg_weights[0] = 0.0  # Do not regularize the bias term
    
    X_reg = np.diag(reg_weights)  # Shape (20, 20)
    
    # Augmented X of shape (N + 20, 20)
    X_aug = np.vstack([X_std_bias, X_reg])
    
    # Augmented Y of shape (N + 20,)
    Y_x_aug = np.concatenate([Y_x, np.zeros(20)])
    Y_y_aug = np.concatenate([Y_y, np.zeros(20)])

    # Solve using standard least-squares (extremely stable)
    x_weights, _, _, _ = np.linalg.lstsq(X_aug, Y_x_aug, rcond=None)
    y_weights, _, _, _ = np.linalg.lstsq(X_aug, Y_y_aug, rcond=None)

    return x_weights, y_weights, means, stds


def predict_gaze(lh, lv, rh, rv, head_features, x_weights, y_weights, means, stds):
    """
    Predicts screen coordinates given current gaze features and standardization maps.
    """
    if x_weights is None or y_weights is None or head_features is None or means is None or stds is None:
        return None

    # head_features structure: (head_tx, head_ty, head_dist, head_yaw, head_pitch)
    h_tx, h_ty, h_dist, h_yaw, h_pitch = head_features

    # 1. Extract 19-D features
    feats = extract_gaze_features(lh, lv, rh, rv, h_yaw, h_pitch, h_dist)

    # 2. Standardize
    feats_std = (feats - means) / stds

    # 3. Add bias coefficient (1.0)
    feats_std_bias = np.insert(feats_std, 0, 1.0)

    # 4. Dot product prediction
    pred_x = np.dot(feats_std_bias, x_weights)
    pred_y = np.dot(feats_std_bias, y_weights)

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

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Calibration states
STATE_PREVIEW = 0
STATE_CALIB_STATIONARY = 1
STATE_CALIB_SHAKING = 2
STATE_CALIBRATED = 3

current_state = STATE_PREVIEW
active_point_idx = 0
calibration_data = []  # Stores: (target_x, target_y, lh, lv, rh, rv, tx, ty, dist, yaw, pitch)

# Ridge coefficients & standardization weights
x_weights = None
y_weights = None
means = None
stds = None

gaze_history = []
HISTORY_SIZE = 6  # Size of smoothing history window
point_start_time = 0.0
point_running = False
point_captured_duration = 0.0
last_time = time.time()
latest_head_features = None
input_smoother = FeatureSmoother(size=5)

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

    latest_lh_ratio = None
    latest_lv_ratio = None
    latest_rh_ratio = None
    latest_rv_ratio = None
    latest_head_features = None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Calculate Head Pose Correction Features
                head_tx = face_landmarks.landmark[1].x
                head_ty = face_landmarks.landmark[1].y
                
                # Eye-corner span Euclidean distance for Z-depth
                dx = face_landmarks.landmark[263].x - face_landmarks.landmark[33].x
                dy = face_landmarks.landmark[263].y - face_landmarks.landmark[33].y
                dz = face_landmarks.landmark[263].z - face_landmarks.landmark[33].z
                head_dist = (dx**2 + dy**2 + dz**2)**0.5
                
                # Yaw horizontal ratio (nose tip relative to outer eye corners)
                head_yaw = (face_landmarks.landmark[1].x - face_landmarks.landmark[33].x) / (face_landmarks.landmark[263].x - face_landmarks.landmark[33].x + 1e-6)
                
                # Pitch vertical ratio (nose tip relative to forehead & chin)
                head_pitch = (face_landmarks.landmark[1].y - face_landmarks.landmark[10].y) / (face_landmarks.landmark[152].y - face_landmarks.landmark[10].y + 1e-6)
                
                
                left_center = get_iris_center(
                    face_landmarks,
                    LEFT_IRIS,
                    w,
                    h
                )

                right_center = get_iris_center(
                    face_landmarks,
                    RIGHT_IRIS,
                    w,
                    h
                )

                cv2.circle(
                    frame,
                    left_center,
                    6,
                    (255, 0, 0),
                    -1
                )

                cv2.circle(
                    frame,
                    right_center,
                    6,
                    (255, 0, 0),
                    -1
                )

                # Left eye
                left_iris_x = left_center[0]
                left_iris_y = left_center[1]

                left_c1_x = int(face_landmarks.landmark[362].x * w)
                left_c1_y = int(face_landmarks.landmark[362].y * h)
                left_c2_x = int(face_landmarks.landmark[263].x * w)
                left_c2_y = int(face_landmarks.landmark[263].y * h)

                left_eye_w_dist = ((left_c2_x - left_c1_x)**2 + (left_c2_y - left_c1_y)**2)**0.5

                left_ratio = calculate_iris_ratio(left_iris_x, left_c1_x, left_c2_x, left_eye_w_dist)

                top_lid_y = int(face_landmarks.landmark[159].y * h)
                bottom_lid_y = int(face_landmarks.landmark[145].y * h)

                left_vertical_ratio = calculate_vertical_ratio(left_iris_y, left_c1_y, left_c2_y, left_eye_w_dist)


                # Right eye
                right_iris_x = right_center[0]
                right_iris_y = right_center[1]

                right_c1_x = int(face_landmarks.landmark[33].x * w)
                right_c1_y = int(face_landmarks.landmark[33].y * h)
                right_c2_x = int(face_landmarks.landmark[133].x * w)
                right_c2_y = int(face_landmarks.landmark[133].y * h)

                right_eye_w_dist = ((right_c2_x - right_c1_x)**2 + (right_c2_y - right_c1_y)**2)**0.5

                right_ratio = calculate_iris_ratio(right_iris_x, right_c1_x, right_c2_x, right_eye_w_dist)

                right_top_lid_y = int(face_landmarks.landmark[386].y * h)
                right_bottom_lid_y = int(face_landmarks.landmark[374].y * h)

                right_vertical_ratio = calculate_vertical_ratio(right_iris_y, right_c1_y, right_c2_y, right_eye_w_dist)
                
                #Average horizontal ratio
                avg_ratio = (
                    left_ratio +
                    right_ratio
                ) / 2
                
                #Average vertical ratio
                avg_vertical_ratio = (
                    left_vertical_ratio +
                    right_vertical_ratio
                ) / 2

                # Calculate Eye Aspect Ratio (EAR) for blink detection
                left_eye_h = abs(top_lid_y - bottom_lid_y)
                left_eye_w = abs(left_c2_x - left_c1_x)
                left_ear = left_eye_h / left_eye_w if left_eye_w > 0 else 0.0

                right_eye_h = abs(right_top_lid_y - right_bottom_lid_y)
                right_eye_w = abs(right_c2_x - right_c1_x)
                right_ear = right_eye_h / right_eye_w if right_eye_w > 0 else 0.0

                # Blink gate: only record and smooth features if both eyes are open
                if left_ear >= 0.18 and right_ear >= 0.18:
                    raw_feats = (left_ratio, left_vertical_ratio, right_ratio, right_vertical_ratio,
                                 head_tx, head_ty, head_dist, head_yaw, head_pitch)
                    sm_feats = input_smoother.add_and_smooth(raw_feats)
                    
                    latest_lh_ratio, latest_lv_ratio, latest_rh_ratio, latest_rv_ratio, sm_tx, sm_ty, sm_dist, sm_yaw, sm_pitch = sm_feats
                    latest_head_features = (sm_tx, sm_ty, sm_dist, sm_yaw, sm_pitch)

                direction = get_gaze_direction(
                    avg_ratio
                )

                if direction == "LEFT":
                    color = (0, 0, 255)      # Red

                elif direction == "RIGHT":
                    color = (255, 0, 0)      # Blue

                else:
                    color = (0, 255, 0)      # Green


                vertical_direction = get_vertical_direction(
                    avg_vertical_ratio
                )

                if vertical_direction == "CENTER":

                    final_direction = direction

                else:

                    final_direction = (
                        vertical_direction +
                        " " +
                        direction
                    )


                cv2.putText(
                    frame,
                    final_direction,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                print(
                    f"H:{avg_ratio:.2f} "
                    f"V:{avg_vertical_ratio:.2f} "
                    f"{final_direction}"
                )

                for idx in LEFT_IRIS + RIGHT_IRIS:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

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
        is_tracking_active = (latest_lh_ratio is not None and latest_head_features is not None)

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

                        # Continuously log data points
                        if is_tracking_active:
                            h_tx, h_ty, h_dist, h_yaw, h_pitch = latest_head_features
                            calibration_data.append((
                                pt[0], pt[1], 
                                latest_lh_ratio, latest_lv_ratio, latest_rh_ratio, latest_rv_ratio,
                                h_tx, h_ty, h_dist, h_yaw, h_pitch
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
        is_tracking_active = (latest_lh_ratio is not None and latest_head_features is not None)

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
                    gaze_history = []
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

                        # Continuously log data points
                        if is_tracking_active:
                            h_tx, h_ty, h_dist, h_yaw, h_pitch = latest_head_features
                            calibration_data.append((
                                pt[0], pt[1], 
                                latest_lh_ratio, latest_lv_ratio, latest_rh_ratio, latest_rv_ratio,
                                h_tx, h_ty, h_dist, h_yaw, h_pitch
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
        # Predict gaze coordinate in real-time
        if latest_lh_ratio is not None and latest_head_features is not None:
            pred = predict_gaze(latest_lh_ratio, latest_lv_ratio, latest_rh_ratio, latest_rv_ratio, 
                                latest_head_features, x_weights, y_weights, means, stds)
            if pred is not None:
                pred_x, pred_y = pred
                pred_x = np.clip(pred_x, 0, canvas_w)
                pred_y = np.clip(pred_y, 0, canvas_h)

                # Smooth gaze predictions to reduce jitter
                gaze_history.append((pred_x, pred_y))
                if len(gaze_history) > HISTORY_SIZE:
                    gaze_history.pop(0)

                smooth_x = int(sum(pt[0] for pt in gaze_history) / len(gaze_history))
                smooth_y = int(sum(pt[1] for pt in gaze_history) / len(gaze_history))

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
            input_smoother.reset()
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
        gaze_history = []
        print("Calibration reset.")


cap.release()
cv2.destroyAllWindows()