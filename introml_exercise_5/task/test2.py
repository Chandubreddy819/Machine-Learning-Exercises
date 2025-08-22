import cv2
import numpy as np
import tensorflow as tf

# Load a pre-trained U-Net model (This is a placeholder; you need to load your actual model)
# model = tf.keras.models.load_model('path_to_your_unet_model.h5')

# Replace with the actual function to load the UNet Model
def create_unet_model():
    # This is just a placeholder function
    return tf.keras.Sequential()

model = create_unet_model()

# Load the image
image_path = r"C:\Users\chand\OneDrive\Pictures\WhatsApp Image 2024-05-29 at 10.45.03_036192b6.jpg"     # Replace with your image path
image = cv2.imread(image_path)

# Preprocess the image
image_resized = cv2.resize(image, (256, 256))  # Resize to match the model's input
image_normalized = image_resized / 255.0  # Normalize the image

# Expand dimensions to add batch size
image_expanded = np.expand_dims(image_normalized, axis=0)

# Make predictions
predictions = model.predict(image_expanded)

# Process predictions
segmented_image = (predictions[0] > 0.5).astype(np.uint8)  # Thresholding
segmented_image = cv2.resize(segmented_image, (image.shape[1], image.shape[0]))  # Resize back to original size

# Display the original and segmented images
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image * 255)  # Scale back for display

cv2.waitKey(0)
cv2.destroyAllWindows()