import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and crop face from an image
def detect_and_crop_face(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # If no faces are detected, return None
    if len(faces) == 0:
        return None
    # Extract the coordinates of the first face detected
    (x, y, w, h) = faces[0]
    crop_img = gray[y:y+h, x:x+w]
    size_img = cv2.resize(crop_img, (220, 220))

    ts = time.time()
    target_file_name = 'faces/' +  ' - ' + str(ts) + '.jpg'
    cv2.imwrite(
        target_file_name,
        size_img,
    )
    return size_img

# Load the images
image1_path = '../dataset/ditya/1703882681 2 in base64_selfie_img.png'
image2_path = '../dataset/rara/User.1.1.jpg'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Detect and crop faces
face1 = detect_and_crop_face(image1, face_cascade)
face2 = detect_and_crop_face(image2, face_cascade)
print('cel')
if face1 is not None and face2 is not None:
    # Resize faces to the same size
    face1_resized = cv2.resize(face1, (100, 100))
    face2_resized = cv2.resize(face2, (100, 100))

    # Flatten the images to 1D arrays for comparison
    face1_vector = face1_resized.flatten()
    face2_vector = face2_resized.flatten()

    # Calculate cosine similarity between the two face vectors
    similarity_score = cosine_similarity([face1_vector], [face2_vector])[0][0]
    similarity_percentage = similarity_score * 100

    # Determine if the faces are similar
    similar = similarity_percentage > 60

    result = {
        "Photo 1": image1_path,
        "Photo 2": image2_path,
        "Similarity Percentage": similarity_percentage,
        "Are Faces Similar": "Yes" if similar else "No"
    }
else:
    result = "Could not detect a face in one or both of the images. Please try with different images."


print (result)