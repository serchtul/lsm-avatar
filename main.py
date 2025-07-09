import cv2
import mediapipe
import os

from dotenv import load_dotenv

load_dotenv()

# The holistic model is considered "legacy" and isn't supported anymore. However, an upgrade to it is yet to be published.
# For more info, check out https://ai.google.dev/edge/mediapipe/solutions/guide#legacy
mp_holistic = mediapipe.solutions.holistic
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles

def main():
    frame_idx = 0
    filename = os.getenv("VIDEO_FILENAME")

    if not filename:
        print("Missing video file")
        exit(1)

    capture = cv2.VideoCapture(filename)

    with mp_holistic.Holistic(
        static_image_mode=False, 
        model_complexity=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.4,
        refine_face_landmarks=True
    ) as holistic:
        while capture.isOpened() and frame_idx < 85: # Stop the video prematurely. This number is arbitrary and can be removed/adjusted
            frame_idx += 1

            success, frame = capture.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)

            mp_drawing.draw_landmarks(frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            mp_drawing.draw_landmarks(frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )
            mp_drawing.draw_landmarks(frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit video
                break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
