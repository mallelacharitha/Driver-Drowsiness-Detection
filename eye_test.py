
import cv2
import mediapipe as mp
import time
import math
import winsound

# ==============================
# SETTINGS
# ==============================

EAR_THRESHOLD = 0.20
CLOSED_TIME = 2.0   # seconds

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==============================
# EAR FUNCTION
# ==============================

def calculate_ear(landmarks, eye_indices, width, height):

    points = []

    for index in eye_indices:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = points

    vertical1 = math.dist(p2, p6)
    vertical2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)

    if horizontal == 0:
        return 0

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear


# ==============================
# MEDIAPIPE SETUP
# ==============================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# ==============================
# CAMERA
# ==============================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

print("===================================")
print(" DRIVER DROWSINESS DETECTION")
print("===================================")
print("Press Q to quit")

closed_start_time = None
alert_active = False

timestamp = 0

# ==============================
# MAIN LOOP
# ==============================

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = camera.read()

        if not ret:
            print("ERROR: Could not read camera")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # ==============================
        # FACE DETECTED
        # ==============================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE,
                width,
                height
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE,
                width,
                height
            )

            average_ear = (left_ear + right_ear) / 2

            # ==============================
            # DISPLAY EAR
            # ==============================

            cv2.putText(
                frame,
                f"LEFT EAR: {left_ear:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"RIGHT EAR: {right_ear:.2f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ==============================
            # EYE STATUS
            # ==============================

            if average_ear < EAR_THRESHOLD:

                cv2.putText(
                    frame,
                    "EYES: CLOSED",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                # Start timer
                if closed_start_time is None:
                    closed_start_time = time.time()

                closed_duration = time.time() - closed_start_time

                cv2.putText(
                    frame,
                    f"CLOSED TIME: {closed_duration:.1f}s",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

                # ==============================
                # DROWSINESS
                # ==============================

                if closed_duration >= CLOSED_TIME:

                    cv2.putText(
                        frame,
                        "DROWSINESS DETECTED!",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3
                    )

                    # Alert only once
                    if not alert_active:

                        winsound.Beep(1200, 500)

                        alert_active = True

            else:

                cv2.putText(
                    frame,
                    "EYES: OPEN",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Reset timer
                closed_start_time = None
                alert_active = False

        else:

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # ==============================
        # SHOW WINDOW
        # ==============================

        cv2.imshow(
            "Driver Drowsiness Detection",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()

print("Program stopped.")

