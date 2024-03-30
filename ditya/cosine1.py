import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model for face detection
model_file = "deploy.prototxt.txt"
weight_file = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(model_file, weight_file)

# Function to detect and crop face from an image
def detect_and_crop_face(image, net):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    # Find the most prominent face in the image
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        
        # Ensure that the detection with the largest confidence also meets our
        # minimum confidence test (thus helping filter out weak detections)
        if confidence < 0.5:
            print('tes')
            return None
        
        # Compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Extract the face ROI and grab the ROI dimensions
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        
        # Ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            print('tes1')
            return None
        else:
            return face
    else:
        print('tes2')
        return None

# Load the images
image1_path = '../dataset/rara/User.1.1.jpg'
image2_path = '../dataset/ika/User.2.37.jpg'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Detect and crop faces
face1 = detect_and_crop_face(image1, face_net)
face2 = detect_and_crop_face(image2, face_net)
print (face1)
print (face2)
if face1 is not None and face2 is not None:
    # Resize faces to the same size for comparison
    face1 = cv2.resize(face1, (100, 100))
    face2 = cv2.resize(face2, (100, 100))

    # Convert faces to grayscale
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

    # Flatten the face images to get the vectors for comparison
    face1_vector = face1_gray.flatten()
    face2_vector = face2_gray.flatten()

    # Compute the cosine similarity between the two vectors
    similarity = cosine_similarity([face1_vector], [face2_vector])

    # Print the similarity score
    similarity_percentage = similarity[0][0] * 100
    print(f"Similarity: {similarity_percentage:.2f}%")

    # Determine if the faces are similar
    if similarity_percentage > 60:
        print("Faces are similar.")
    else:
        print("Faces are not similar.")
else:
    print("Could not detect a face in one or both of the images. Please try with different images.")

