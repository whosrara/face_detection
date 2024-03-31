import os
import random

# Assuming cosine5.py is correctly set up with a compare_images function
from ssim1 import compare_images

# Find Threshold
from findthreshold import find_best_threshold

# Path to the dataset directory
dataset_path = './dataset'

def run_test(dataset_path, iterations=10):
    i = 0
    r = []

    for _ in range(iterations):
        # Randomly decide if the images should come from the same folder or different ones
        same_folder = random.choice([True, False])
        
        # Get the list of person directories in the dataset
        persons = [person for person in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, person))]

        if same_folder:
            # Randomly select a person's directory and use it for both images
            selected_person = random.choice(persons)
            person_paths = [os.path.join(dataset_path, selected_person), os.path.join(dataset_path, selected_person)]
        else:
            # Randomly select two person's directories for the two images
            selected_persons = random.sample(persons, 2) if len(persons) >= 2 else [random.choice(persons), random.choice(persons)]
            person_paths = [os.path.join(dataset_path, person) for person in selected_persons]

        images = []
        for person_path in person_paths:
            # Get the list of image files in the person's directory
            person_images = [img for img in os.listdir(person_path) if os.path.isfile(os.path.join(person_path, img))]
            # Randomly select an image from the person's directory
            images.append(random.choice(person_images))

        # Construct the full image paths
        image_paths = [os.path.join(person_paths[i], images[i]) for i in range(2)]

        # Compare the images and print the result
        similarity_result = compare_images(image_paths[0], image_paths[1])
        # print(f"Test iteration: Comparing images {images[0]} and {images[1]} {'from the same folder' if same_folder else 'from different folders'}")
        # label = ''
        # if same_folder: label += 'Yes - '
        # else: label += 'No - '

        # label += image_paths[0]
        # label += ' / '
        # label += image_paths[1]
        # print(label)
        
        # print(similarity_result)
        # print("\n")
        predict_result = 'YES' if same_folder else 'NO'
        actual_result = similarity_result.get("Are Faces Similar")
        error = similarity_result.get('Error')
        photo_1 = similarity_result.get('Photo 1')
        photo_2 = similarity_result.get('Photo 2')

        score = similarity_result.get('Similarity Percentage')

        label = image_paths[0] + ',' + image_paths[1] + ',' + str(photo_1) + ',' + str(photo_2) + ',' + str(score) + ',' + str(predict_result) + ',' + str(actual_result) + ',' + str(error)
        print(label)
        # r[i]['score_similarity'] = score
        # r[i]['predict_result'] = predict_result
        r.append({'score_similarity': 0 if score == None else score, 'predict_result': predict_result})
        i = i+1

    print(r)
    best_threshold, best_accuracy = find_best_threshold(r)
    print(f"Best Threshold: {best_threshold}, Best Accuracy: {best_accuracy}")

# Specify how many iterations of the test you want to run
number_of_tests = 5  # For example, 5 iterations
run_test(dataset_path, iterations=number_of_tests)
