import cv2
import mediapipe as mp
import numpy as np



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

def calculate_iris_ratio(iris_x, left_corner_x,right_corner_x):

    eye_width = abs(right_corner_x - left_corner_x)

    if eye_width == 0:
        return 0

    ratio = (iris_x - min(left_corner_x, right_corner_x)) / eye_width

    return ratio

def calculate_vertical_ratio(iris_y,top_lid_y,bottom_lid_y):

    eye_height = abs(bottom_lid_y - top_lid_y)

    if eye_height == 0:
        return 0

    ratio = (iris_y - min(top_lid_y, bottom_lid_y)) / eye_height

    return ratio


def calibrate_gaze(calibration_data):
    """
    Fits polynomial or linear regression mappings between calibration gaze ratios and screen coordinates.
    """
    if len(calibration_data) < 6:
        # Linear Model: target_x = a0 + a1*h + a2*v
        X, Y, A = [], [], []
        for tx, ty, h, v in calibration_data:
            X.append(tx)
            Y.append(ty)
            A.append([1, h, v])
        
        A = np.array(A)
        X = np.array(X)
        Y = np.array(Y)
        
        x_coeffs, _, _, _ = np.linalg.lstsq(A, X, rcond=None)
        y_coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        return x_coeffs, y_coeffs, False
        
    # Second-order Polynomial Model
    X, Y, A = [], [], []
    for tx, ty, h, v in calibration_data:
        X.append(tx)
        Y.append(ty)
        A.append([1, h, v, h**2, v**2, h*v])
        
    A = np.array(A)
    X = np.array(X)
    Y = np.array(Y)
    
    x_coeffs, _, _, _ = np.linalg.lstsq(A, X, rcond=None)
    y_coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    return x_coeffs, y_coeffs, True


def predict_gaze(h, v, x_coeffs, y_coeffs, is_poly):
    """
    Predicts screen coordinates given horizontal and vertical eye ratios.
    """
    if x_coeffs is None or y_coeffs is None:
        return None
        
    if is_poly:
        px = x_coeffs[0] + x_coeffs[1]*h + x_coeffs[2]*v + x_coeffs[3]*(h**2) + x_coeffs[4]*(v**2) + x_coeffs[5]*(h*v)
        py = y_coeffs[0] + y_coeffs[1]*h + y_coeffs[2]*v + y_coeffs[3]*(h**2) + y_coeffs[4]*(v**2) + y_coeffs[5]*(h*v)
    else:
        px = x_coeffs[0] + x_coeffs[1]*h + x_coeffs[2]*v
        py = y_coeffs[0] + y_coeffs[1]*h + y_coeffs[2]*v
        
    return int(px), int(py)

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Calibration states & data
STATE_PREVIEW = 0
STATE_CALIBRATING = 1
STATE_CALIBRATED = 2

current_state = STATE_PREVIEW
active_point_idx = 0
calibration_data = []  # Stores: (target_x, target_y, h_ratio, v_ratio)
x_coeffs = None
y_coeffs = None
is_poly = False
gaze_history = []
HISTORY_SIZE = 6  # Size of smoothing history window

# 9 calibration points mapped to a 1280x720 canvas
CALIBRATION_POINTS = [
    (128, 72),   # Top-Left
    (640, 72),   # Top-Center
    (1152, 72),  # Top-Right
    (128, 360),  # Middle-Left
    (640, 360),  # Center
    (1152, 360), # Middle-Right
    (128, 648),  # Bottom-Left
    (640, 648),  # Bottom-Center
    (1152, 648)  # Bottom-Right
]

while True:

    success, frame = cap.read()

    if not success:
        break

    latest_h_ratio = None
    latest_v_ratio = None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
        
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

                left_eye_corner_1 = int(face_landmarks.landmark[362].x * w)
                left_eye_corner_2 = int(face_landmarks.landmark[263].x * w)

                left_ratio = calculate_iris_ratio(left_iris_x, left_eye_corner_1, left_eye_corner_2)

                left_iris_y = left_center[1]

                top_lid_y = int(face_landmarks.landmark[159].y * h)
                bottom_lid_y = int(face_landmarks.landmark[145].y * h)

                left_vertical_ratio = calculate_vertical_ratio(left_iris_y, top_lid_y, bottom_lid_y)


                # Right eye
                right_iris_x = right_center[0]

                right_eye_corner_1 = int(face_landmarks.landmark[33].x * w)
                right_eye_corner_2 = int(face_landmarks.landmark[133].x * w)

                right_ratio = calculate_iris_ratio(right_iris_x, right_eye_corner_1, right_eye_corner_2)

                right_iris_y = right_center[1]

                right_top_lid_y = int(face_landmarks.landmark[386].y * h)

                right_bottom_lid_y = int(face_landmarks.landmark[374].y * h)

                right_vertical_ratio = calculate_vertical_ratio(right_iris_y, right_top_lid_y, right_bottom_lid_y)
                
                #Average horizontal ratio
                avg_ratio = (
                    left_ratio +
                    right_ratio
                ) / 2
                latest_h_ratio = avg_ratio

                #Average vertical ratio
                avg_vertical_ratio = (
                    left_vertical_ratio +
                    right_vertical_ratio
                ) / 2
                latest_v_ratio = avg_vertical_ratio

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

    # Construct calibration & visualization canvas (1280x720)
    canvas_w, canvas_h = 1280, 720
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

    elif current_state == STATE_CALIBRATING:
        # Render 9-point layout
        for idx, pt in enumerate(CALIBRATION_POINTS):
            if idx == active_point_idx:
                # Active target point
                cv2.circle(canvas, pt, 24, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, pt, 8, (0, 140, 255), -1, cv2.LINE_AA)
                cv2.line(canvas, (pt[0]-35, pt[1]), (pt[0]+35, pt[1]), (0, 140, 255), 1)
                cv2.line(canvas, (pt[0], pt[1]-35), (pt[0], pt[1]+35), (0, 140, 255), 1)
            else:
                # Faint uncalibrated point
                cv2.circle(canvas, pt, 6, (80, 80, 80), -1, cv2.LINE_AA)

        # Minimal Instructions at bottom center
        cv2.putText(canvas, "Look at the orange target and press SPACE.", (360, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

    elif current_state == STATE_CALIBRATED:
        # Predict gaze coordinate in real-time
        if latest_h_ratio is not None and latest_v_ratio is not None:
            pred = predict_gaze(latest_h_ratio, latest_v_ratio, x_coeffs, y_coeffs, is_poly)
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

        # Status text
        cv2.putText(canvas, "Gaze Tracking Active. Press 'R' to recalibrate.", (40, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

    # Show window
    cv2.imshow("AffectIQ Eye-Tracking", canvas)

    # Keyboard control handler
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c') or key == ord('C'):
        if current_state == STATE_PREVIEW:
            current_state = STATE_CALIBRATING
            active_point_idx = 0
            calibration_data = []
            x_coeffs = None
            y_coeffs = None
            print("Calibration sequence initiated.")
    elif key == ord('r') or key == ord('R'):
        current_state = STATE_PREVIEW
        active_point_idx = 0
        calibration_data = []
        x_coeffs = None
        y_coeffs = None
        gaze_history = []
        print("Calibration reset.")
    elif key == 32:  # SPACE
        if current_state == STATE_CALIBRATING and latest_h_ratio is not None and latest_v_ratio is not None:
            target_x, target_y = CALIBRATION_POINTS[active_point_idx]
            calibration_data.append((target_x, target_y, latest_h_ratio, latest_v_ratio))
            print(f"Recorded point {active_point_idx+1}/9: screen=({target_x}, {target_y}), gaze=({latest_h_ratio:.4f}, {latest_v_ratio:.4f})")
            active_point_idx += 1
            if active_point_idx >= len(CALIBRATION_POINTS):
                # Calculate calibration coefficients
                x_coeffs, y_coeffs, is_poly = calibrate_gaze(calibration_data)
                current_state = STATE_CALIBRATED
                gaze_history = []
                print("Calibration complete. Gaze coefficients computed.")


cap.release()
cv2.destroyAllWindows()