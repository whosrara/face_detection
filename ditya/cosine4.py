import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# Initialize Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and crop the face from an image
def detect_and_crop_face(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return image[y:y+h, x:x+w]

# Load the images
# Tes 1
# image1_path = '../dataset/ditya/0106.png'
# image2_path = '../dataset/zidni/0109.png'

# Tes 2
# image1_path = '../dataset/rara/User.1.1.jpg'
# image2_path = '../dataset/rara/User.1.3.jpg'

# Tes 3
# image1_path = '../dataset/ika/User.2.5.jpg'
# image2_path = '../dataset/ika/User.2.37.jpg'

# Tes 4
# image1_path = '../dataset/rara/User.1.5.jpg'
# image2_path = '../dataset/ika/User.2.37.jpg'

# Tes 5
image1_path = '../dataset/ditya/0111.png'
image2_path = '../dataset/ditya/0109.png'

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Detect and crop faces
face1 = detect_and_crop_face(image1, face_cascade)
face2 = detect_and_crop_face(image2, face_cascade)

# Function to preprocess and compute SSIM
def compute_ssim(face1, face2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    # Resize images to the same size
    gray1 = cv2.resize(gray1, (100, 100))
    gray2 = cv2.resize(gray2, (100, 100))
    # Compute SSIM between two images
    score, _ = compare_ssim(gray1, gray2, full=True)
    return score

# Compare the faces if both were detected
if face1 is not None and face2 is not None:
    ssim_score = compute_ssim(face1, face2)
    similarity_percentage = ssim_score * 100
    result = {
        "Similarity Percentage": similarity_percentage,
        "Are Faces Similar": "Yes" if similarity_percentage > 60 else "No"
    }
else:
    result = {
        "Error": "Could not detect a face in one or both of the images. Please try with different images."
    }

# Print the result
print(result)
