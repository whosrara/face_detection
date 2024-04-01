import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("./faces.npy")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./video.mp4')

# Initialize Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier(4)
model.fit(X, y)

def detect_and_crop_face(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return image[y:y+h, x:x+w]

# image1_path = './dataset/ika/User.2.1.jpg'
# image1_path = './dataset/ditya/0111.png'
image1_path = './dataset/rara/User.1.1.jpg'
image1 = cv2.imread(image1_path)
face1 = detect_and_crop_face(image1, face_cascade)

gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.resize(gray1, (100, 100))
gray1 = gray1.flatten()
res = model.predict([gray1])
print(res)
