import cv2
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and crop the face from an image
def detect_and_crop_face(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # If no faces are detected, return None
    if len(faces) == 0:
        return None
    # Extract the coordinates of the first face detected
    (x, y, w, h) = faces[0]
    return image[y:y+h, x:x+w]

# Function to apply Canny edge detection on a face
def get_edge_map(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edge_map = cv2.Canny(gray_face, 100, 200)
    return edge_map

# Function to compare edge maps of the faces
def compare_edge_maps(face1, face2):
    edge_map1 = get_edge_map(face1)
    edge_map2 = get_edge_map(face2)

    # Resize the edge maps to a common size
    edge_map1_resized = cv2.resize(edge_map1, (100, 100))
    edge_map2_resized = cv2.resize(edge_map2, (100, 100))

    # Flatten the edge maps to 1D arrays for comparison
    edge_map1_vector = edge_map1_resized.flatten()
    edge_map2_vector = edge_map2_resized.flatten()

    # Calculate the cosine similarity between the two edge map vectors
    similarity_score = cosine_similarity([edge_map1_vector], [edge_map2_vector])[0][0]
    similarity_percentage = similarity_score * 100

    return similarity_percentage

# Load the images
image1_path = '../dataset/ditya/0106.png'
image2_path = '../dataset/ditya/0109.png'
# image1_path = '../dataset/rara/User.1.5.jpg'
# image2_path = '../dataset/rara/User.1.1.jpg'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Detect and crop faces
face1 = detect_and_crop_face(image1, face_cascade)
face2 = detect_and_crop_face(image2, face_cascade)

# Results variable
result = {}

# Compare the edge maps of the two faces if both were detected
if face1 is not None and face2 is not None:
    similarity_percentage = compare_edge_maps(face1, face2)
    result["Similarity Percentage"] = similarity_percentage
    result["Are Faces Similar"] = "Yes" if similarity_percentage > 60 else "No"
else:
    result["Error"] = "Could not detect a face in one or both of the images. Please try with different images."

# Print the result
print(result)
