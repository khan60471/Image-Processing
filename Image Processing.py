import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(images, titles):
    """Helper function to display multiple images in one figure."""
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()

# Load an image
image_path = 'Cat2.jpg'  # Update this path to your image file
image = cv2.imread(image_path)

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur (Filtering)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Canny Edge Detection
edges = cv2.Canny(blurred_image, 100, 200)

# Apply Image Transformation: Rotate the image
(center_x, center_y) = (gray_image.shape[1] // 2, gray_image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), 45, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# Display images
images = [image, gray_image, blurred_image, edges, rotated_image]
titles = ['Original Image', 'Gray Image', 'Blurred Image', 'Edges', 'Rotated Image']
display_images(images, titles)
