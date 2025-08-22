import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Read the image
image_path = r'C:\Users\chand\OneDrive\Pictures\Camera Roll\WIN_20240911_12_43_35_Pro.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Convert the BGR image to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform pose estimation
result = pose.process(rgb_image)

# Draw the pose annotation on the image
if result.pose_landmarks:
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS)

# Display the image with pose landmarks
cv2.imshow('Pose Estimation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()