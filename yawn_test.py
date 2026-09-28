import cv2
import mediapipe as mp
import math

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

print("===================================")
print(" YAWNING DETECTION TEST")
print("===================================")
print("Keep mouth closed first.")
print("Then open your mouth widely.")
print("Press Q to quit.")

timestamp = 0

# Mouth landmarks
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

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

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # Get mouth points
            upper = landmarks[UPPER_LIP]
            lower = landmarks[LOWER_LIP]
            left = landmarks[LEFT_MOUTH]
            right = landmarks[RIGHT_MOUTH]

            upper_x = upper.x * width
            upper_y = upper.y * height

            lower_x = lower.x * width
            lower_y = lower.y * height

            left_x = left.x * width
            left_y = left.y * height

            right_x = right.x * width
            right_y = right.y * height

            # Mouth opening
            vertical_distance = math.dist(
                (upper_x, upper_y),
                (lower_x, lower_y)
            )

            # Mouth width
            horizontal_distance = math.dist(
                (left_x, left_y),
                (right_x, right_y)
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
                f"Mouth Ratio: {mouth_ratio:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Initial threshold
            if mouth_ratio > 0.60:

                cv2.putText(
                    frame,
                    "YAWNING / MOUTH OPEN",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "MOUTH NORMAL",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
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

        cv2.imshow(
            "Yawning Detection Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

print("Program stopped.")