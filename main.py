import cv2
import mediapipe as mp


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

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

while True:

    success, frame = cap.read()

    if not success:
        break

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

                #Average vertical ratio
                avg_vertical_ratio = (
                    left_vertical_ratio +
                    right_vertical_ratio
                ) / 2

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

    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()