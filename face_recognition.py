import cv2
import numpy as np
import os

# Fungsi untuk memuat dataset wajah
def load_dataset(dataset_path):
    face_data = []
    labels = []
    names = {}
    class_id = 0

    for fx in os.listdir(dataset_path):
        if fx.endswith('.jpg'):
            names[class_id] = fx[:-4]
            img = cv2.imread(dataset_path + fx, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100)).astype(np.float32)  # Mengubah ukuran dan tipe data gambar
            face_data.append(img)
            target = class_id * np.ones((1,))
            labels.append(target)
            class_id += 1

    face_data = np.array(face_data)  # Tanpa spesifikasi tipe data
    labels = np.array(labels, dtype=np.int32)  # Mengonversi ke tipe data int32
    return face_data, labels, names


# Fungsi untuk klasifikasi menggunakan KNN
def knn_classification(train_data, train_labels, test_data, k=5):
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    ret, results, neighbours, dist = knn.findNearest(test_data, k=k)
    return results.flatten().astype(int)

# Fungsi untuk menggambar kotak dan nama pada wajah yang terdeteksi
def draw_text(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Path dataset wajah
dataset_path = "./face_dataset/"

# Memuat dataset wajah
face_data, labels, names = load_dataset(dataset_path)

# Inisialisasi detektor wajah
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Mulai kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Ekstraksi region of interest (ROI)
        roi = gray_frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (100, 100))

        # Klasifikasi wajah menggunakan KNN
        result = knn_classification(face_data.reshape(-1, 100*100), labels.flatten(), roi.flatten())

        # Menentukan label dan nama
        label = result[0]
        name = names[label]

        # Menggambar kotak dan nama pada wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        draw_text(frame, name, x, y-10)

    # Menampilkan frame
    cv2.imshow("Face Recognition", frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import os

# #KNN
# def distance (v1, v2):
#     return np.sqrt(((v1-v2)**2).sum())

# def knn(train, test, k=5):
#     dist = []

#     for i in range(train.shape[0]):
#         ix = train [i, :-1]
#         iy = train [i, -1]
#         d = distance(test, ix)
#         dist.append([d, iy])
#     dk = sorted(dist, key = lambda x: x[0])[:k]
#     labels = np.array(dk)[:, -1]

#     output = np.unique(labels, return_counts=True)
#     index = np.argmax(output[1])
#     return output[0][index]

# cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# dataset_path = "./face_dataset/"

# face_data = []
# labels = []
# class_id = 0
# names = {}

# # dataset preparation
# for fx in os.listdir(dataset_path):
#     if fx.endswith('.npy'):
#         names[class_id] = fx[:-4]
#         data_item = np.load(dataset_path +fx)
#         face_data.append(data_item)

#         target = class_id *np.ones((data_item.shape[0],))
#         class_id += 1
#         labels.append(target)

# face_dataset = np.concatenate(face_data, axis=0)
# face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
# print(face_labels.shape)
# print(face_dataset.shape)

# trainset = np.concatenate((face_dataset, face_labels), axis=1)
# print(trainset.shape)

# font = cv2.FONT_HERSHEY_SIMPLEX

# while True:
#     ret, frame = cap.read()
#     if ret == False:
#         continue

#     # frame grayscale
#     gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect multi face
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for face in faces:
#         x, y, w, h = face

#         # face ROI
#         offset = 5
#         face_offset = frame[y-offset: y+h+offset, x-offset: x+w+offset]
#         face_selection = cv2.resize(face_offset, (100,100))

#         out = knn(trainset, face_selection.flatten())

#         cv2.putText(frame, names[int(out)], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

#     cv2.imshow("faces", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()
