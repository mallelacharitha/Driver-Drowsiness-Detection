import cv2
import mediapipe as mp
import time
import math
import winsound

# ============================================================
# SETTINGS
# ============================================================

EAR_THRESHOLD = 0.20
CLOSED_TIME = 2.0

HEAD_THRESHOLD = 0.70

YAWN_THRESHOLD = 0.60

# ============================================================
# EYE LANDMARKS
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ============================================================
# MOUTH LANDMARKS
# ============================================================

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# ============================================================
# EAR FUNCTION
# ============================================================

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


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened")
    exit()

print("==============================================")
print(" DRIVER DROWSINESS DETECTION SYSTEM")
print("==============================================")
print("Eyes closed : 2 seconds")
print("Head down   : threshold 0.70")
print("Yawning     : threshold 0.60")
print("Alert       : ON")
print("Press Q to quit")
print("==============================================")

# ============================================================
# VARIABLES
# ============================================================

closed_start_time = None

# Prevent continuous beeping every frame
last_alert_time = 0

ALERT_COOLDOWN = 3.0

timestamp = 0

# ============================================================
# MAIN PROGRAM
# ============================================================

with mp.tasks.vision.FaceLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        ret, frame = camera.read()

        if not ret:

            print("ERROR: Could not read camera")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # ====================================================
        # DEFAULT STATUS
        # ====================================================

        eye_status = "OPEN"
        head_status = "NORMAL"
        yawn_status = "NORMAL"

        eye_drowsy = False
        head_drowsy = False
        yawn_drowsy = False

        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # =================================================
            # 1. EYE DETECTION
            # =================================================

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

            average_ear = (
                left_ear + right_ear
            ) / 2.0

            # Display EAR
            cv2.putText(
                frame,
                f"LEFT EAR: {left_ear:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"RIGHT EAR: {right_ear:.2f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # Eye status
            if average_ear < EAR_THRESHOLD:

                eye_status = "CLOSED"

                if closed_start_time is None:

                    closed_start_time = time.time()

                closed_duration = (
                    time.time() - closed_start_time
                )

                cv2.putText(
                    frame,
                    f"EYE CLOSED TIME: {closed_duration:.1f}s",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2
                )

                if closed_duration >= CLOSED_TIME:

                    eye_drowsy = True

            else:

                eye_status = "OPEN"

                closed_start_time = None

                cv2.putText(
                    frame,
                    "EYE CLOSED TIME: 0.0s",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            # =================================================
            # 2. HEAD DOWN DETECTION
            # =================================================

            nose = landmarks[1]
            forehead = landmarks[10]
            chin = landmarks[152]

            nose_y = nose.y * height
            forehead_y = forehead.y * height
            chin_y = chin.y * height

            face_height = abs(
                chin_y - forehead_y
            )

            if face_height > 0:

                relative_nose = (
                    nose_y - forehead_y
                ) / face_height

            else:

                relative_nose = 0

            cv2.putText(
                frame,
                f"HEAD VALUE: {relative_nose:.2f}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            if relative_nose > HEAD_THRESHOLD:

                head_status = "DOWN"
                head_drowsy = True

            else:

                head_status = "NORMAL"

            # =================================================
            # 3. YAWNING DETECTION
            # =================================================

            upper = landmarks[UPPER_LIP]
            lower = landmarks[LOWER_LIP]
            left = landmarks[LEFT_MOUTH]
            right = landmarks[RIGHT_MOUTH]

            upper_point = (
                upper.x * width,
                upper.y * height
            )

            lower_point = (
                lower.x * width,
                lower.y * height
            )

            left_point = (
                left.x * width,
                left.y * height
            )

            right_point = (
                right.x * width,
                right.y * height
            )

            vertical_distance = math.dist(
                upper_point,
                lower_point
            )

            horizontal_distance = math.dist(
                left_point,
                right_point
            )

            if horizontal_distance > 0:

                mouth_ratio = (
                    vertical_distance /
                    horizontal_distance
                )

            else:

                mouth_ratio = 0

            cv2.putText(
                frame,
                f"MOUTH RATIO: {mouth_ratio:.2f}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            if mouth_ratio > YAWN_THRESHOLD:

                yawn_status = "YAWNING"
                yawn_drowsy = True

            else:

                yawn_status = "NORMAL"

            # =================================================
            # STATUS DISPLAY
            # =================================================

            cv2.putText(
                frame,
                f"EYES: {eye_status}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if eye_status == "CLOSED"
                else (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"HEAD: {head_status}",
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if head_status == "DOWN"
                else (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"MOUTH: {yawn_status}",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if yawn_status == "YAWNING"
                else (0, 255, 0),
                2
            )

            # =================================================
            # FINAL DROWSINESS DECISION
            # =================================================

            if eye_drowsy or head_drowsy or yawn_drowsy:

                reasons = []

                if eye_drowsy:
                    reasons.append("EYES CLOSED")

                if head_drowsy:
                    reasons.append("HEAD DOWN")

                if yawn_drowsy:
                    reasons.append("YAWNING")

                reason_text = " + ".join(reasons)

                # Alert message
                cv2.putText(
                    frame,
                    "!!! DROWSINESS ALERT !!!",
                    (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    reason_text,
                    (20, 365),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                # =================================================
                # ALERT SOUND
                # =================================================

                current_time = time.time()

                if (
                    current_time - last_alert_time
                    >= ALERT_COOLDOWN
                ):

                    winsound.Beep(1200, 700)

                    last_alert_time = current_time

            else:

                cv2.putText(
                    frame,
                    "STATUS: NORMAL",
                    (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    3
                )

        else:

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Driver Drowsiness Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("Program stopped.")