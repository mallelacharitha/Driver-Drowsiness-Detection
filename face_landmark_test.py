import cv2
import mediapipe as mp
import time

# MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# Configure face landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

print("Face landmark detection started")
print("Press Q to stop")

start_time = time.time()

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = camera.read()

        if not ret:
            print("ERROR: Could not read camera")
            break

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp must increase
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Detect face landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Draw landmark points
        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                h, w, _ = frame.shape

                for landmark in face_landmarks:

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                "FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("MediaPipe Face Landmark Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()