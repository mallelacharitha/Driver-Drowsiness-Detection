import cv2
import mediapipe as mp

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

print("Head position test started")
print("Keep your head NORMAL, then look DOWN")
print("Press Q to quit")

timestamp = 0

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

            # Nose position
            nose = landmarks[1]

            nose_x = nose.x * width
            nose_y = nose.y * height

            # Face center using forehead and chin
            forehead = landmarks[10]
            chin = landmarks[152]

            forehead_y = forehead.y * height
            chin_y = chin.y * height

            face_height = abs(chin_y - forehead_y)

            # Calculate relative nose position
            relative_nose = (nose_y - forehead_y) / face_height

            cv2.putText(
                frame,
                f"Nose Y: {relative_nose:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Simple head-position indication
            if relative_nose > 0.70:

                status = "HEAD DOWN"

                cv2.putText(
                    frame,
                    status,
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            else:

                status = "HEAD NORMAL"

                cv2.putText(
                    frame,
                    status,
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
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

        cv2.imshow("Head Position Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()