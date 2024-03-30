import cv2
import numpy as np
import time

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

# Custom SSIM computation function
def compute_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))

    target_file_name1 = 'faces/' +  ' - ' + str(time.time()) + '.jpg'
    cv2.imwrite(
        target_file_name1,
        img1,
    )
    target_file_name2 = 'faces/' +  ' - ' + str(time.time()) + '.jpg'
    cv2.imwrite(
        target_file_name2,
        img2,
    )

    mean_img1 = np.mean(img1)
    mean_img2 = np.mean(img2)
    var_img1 = np.var(img1)
    var_img2 = np.var(img2)
    std_img1 = np.sqrt(var_img1)
    std_img2 = np.sqrt(var_img2)
    covariance = np.mean((img1 - mean_img1) * (img2 - mean_img2))
    numerator = (2 * mean_img1 * mean_img2 + C1) * (2 * covariance + C2)
    denominator = (mean_img1 ** 2 + mean_img2 ** 2 + C1) * (var_img1 + var_img2 + C2)
    ssim = numerator / denominator
    return ssim


def compare_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    face1 = detect_and_crop_face(image1, face_cascade)
    face2 = detect_and_crop_face(image2, face_cascade)
    # Results variable
    result = {}

    # Compare the faces if both were detected and cropped
    if face1 is not None and face2 is not None:
        ssim_index = compute_ssim(face1, face2)
        similarity_percentage = ssim_index * 100
        result["Similarity Percentage"] = similarity_percentage
        result["Are Faces Similar"] = "Yes" if similarity_percentage > 50 else "No"
    else:
        result["Error"] = "Could not detect a face in one or both of the images. Please try with different images."
        result['Photo 1'] = 'Exist' if face1 is not None else 'Not Exist'
        result['Photo 2'] = 'Exist' if face2 is not None else 'Not Exist'
    return result



# =================================================================
# INTERNAL TEST
# =================================================================

# # Load the images
# # Tes 1
# image1_path = '../dataset/ditya/0106.png'
# image2_path = '../dataset/zidni/0109.png'

# # Tes 2
# # image1_path = '../dataset/rara/User.1.1.jpg'
# # image2_path = '../dataset/rara/User.1.3.jpg'

# # Tes 3
# # image1_path = '../dataset/ika/User.2.5.jpg'
# # image2_path = '../dataset/ika/User.2.37.jpg'

# # Tes 4
# # image1_path = '../dataset/rara/User.1.5.jpg'
# # image2_path = '../dataset/ika/User.2.37.jpg'

# # Tes 5
# # image1_path = '../dataset/ditya/0104.png'
# # image2_path = '../dataset/ditya/0105.png'

# image1 = cv2.imread(image1_path)
# image2 = cv2.imread(image2_path)

# # Detect and crop faces
# face1 = detect_and_crop_face(image1, face_cascade)
# face2 = detect_and_crop_face(image2, face_cascade)

# # Results variable
# result = {}

# # Compare the faces if both were detected and cropped
# if face1 is not None and face2 is not None:
#     ssim_index = compute_ssim(face1, face2)
#     similarity_percentage = ssim_index * 100
#     result["Similarity Percentage"] = similarity_percentage
#     result["Are Faces Similar"] = "Yes" if similarity_percentage > 50 else "No"
# else:
#     result["Error"] = "Could not detect a face in one or both of the images. Please try with different images."

# # Print the result
# print(result)
