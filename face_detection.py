import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    #deteksi wajah
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    for face in faces[:1]:
        x, y, w, h = face

        offset = 10
        face_offset = frame[y-offset: y+h+offset, x-offset: x+w+offset]
        face_selection = cv2.resize(face_offset, (100,100))

        cv2.imshow("Face", face_selection)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)

        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Deteksi mata dan kacamata dalam ROI wajah
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            # Gambar kotak di sekitar mata atau kacamata
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    cv2.imshow("faces", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord ('q'):
        break

cap.release()
cv2.destroyAllWindows()