import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os

facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the directory containing the face images
data_dir = 'face_dataset/'

# Initialize lists to store face data and labels
faces = []
labels = []

# Loop through each image in the directory
for filename in os.listdir(data_dir):
    # Construct the full path to the image file
    filepath = os.path.join(data_dir, filename)

    # Read the image using OpenCV
    img = cv2.imread(filepath)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)

    # Assuming each image contains exactly one face
    for (a, b, w, h) in face_coordinates:
        # Extract the face region from the image
        face = gray[b:b + h, a:a + w]

        # Resize the face region to a fixed size (e.g., 50x50 pixels)
        resized_face = cv2.resize(face, (50, 50))

        # Flatten and reshape the resized face to a 1D array
        flattened_face = resized_face.flatten().reshape(-1)

        # Append the flattened face to the faces list
        faces.append(flattened_face)

        # Assuming labels are derived from image filenames (e.g., filename is the name of the person)
        label = filename.split('.')[0]  # Assuming filename format is 'person1.jpg', 'person2.jpg', etc.
        labels.append(label)

# Convert the lists of face data and labels to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

camera = cv2.VideoCapture(0)

print('Shape of Faces matrix --> ', faces.shape)
knn = KNeighborsClassifier(n_neighbors=4)
print (len(faces), len(labels))
knn.fit(faces, labels)

while True:
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)

        for (a, b, w, h) in face_coordinates:
            fc = gray[b:b + h, a:a + w]
            resized_fc = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            text = knn.predict(resized_fc)
            cv2.putText(frame, text[0], (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (a, b), (a + w, b + w), (0, 0, 255), 2)

        cv2.imshow('livetime face recognition', frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("error")
        break

cv2.destroyAllWindows()
camera.release()
