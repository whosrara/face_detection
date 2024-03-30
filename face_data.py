import cv2
import time

# camera
cap = cv2.VideoCapture(0)
# haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# meminta id
id = input("nama : ")
a = 0

while True:
    a = a+1
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    print(faces)

    for (x, y, w, h) in faces:
        # membuat file foto
        cv2.imwrite('face_dataset/User.'+str(id)+'.'+str(a)+'.jpg', gray_frame[y: y+h, x: x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("face data", frame)
    time.sleep(0.2)
    # mengambil 50 foto
    if (a > 49):
        break

        # Menghentikan jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# skip = 0
# face_data = []
# dataset_path = "./face_dataset/"

# file_name = input("masukkan nama : ")

# while True:
#     ret, frame = cap.read()
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     if ret == False:
#         continue

#     faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
#     if len(faces) == 0:
#         continue

#     k = 1

#     faces = sorted(faces, key=lambda x : x[2]*x[3], reverse=True)
#     skip +=1
#     for face in faces[:1]:
#         x,y,w,h = face

#         offset = 5
#         face_offset = frame[y-offset: y+h+offset, x-offset: x+w+offset]
#         face_selection = cv2.resize(face_offset, (100,100))

#         if skip % 10 == 0:
#             face_data.append(face_selection)
#             print (len(face_data))

#         cv2.imshow(str(k), face_selection)
#         k += 1

#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
#     cv2.imshow("video frame", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# face_data = np.array(face_data)
# face_data = face_data.reshape((face_data.shape[0], -1))
# print (face_data.shape)

# np.save(dataset_path + file_name, face_data)
# print ("dataset save : {}".format(dataset_path +file_name + '.npy'))

# cap.release()
# cv2.destroyAllWindows()