# r = [
#     {"score_similarity": 14.16, "predict_result": "NO"},
#     {"score_similarity": 82.16, "predict_result": "YES"},
#     {"score_similarity": 47.68, "predict_result": "NO"},
# ]


# Assuming predict_result is the ground truth, we find the best threshold for score_similarity
def find_best_threshold(results):

    # Convert score_similarity to float if they aren't
    for item in results:
        item["score_similarity"] = float(item["score_similarity"])
        
    # Extract all score_similarity values
    scores = [item["score_similarity"] for item in results]
    
    # Generate possible thresholds (midpoints between successive scores when sorted)
    scores.sort()
    thresholds = [(scores[i] + scores[i + 1]) / 2 for i in range(len(scores) - 1)]
    
    best_threshold = None
    best_accuracy = 0
    
    # Test each threshold
    for threshold in thresholds:
        correct_predictions = sum(
            (item["predict_result"] == "YES" and item["score_similarity"] > threshold) or
            (item["predict_result"] == "NO" and item["score_similarity"] <= threshold)
            for item in results
        )
        accuracy = correct_predictions / len(results)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy

# best_threshold, best_accuracy = find_best_threshold(r)
# print(f"Best Threshold: {best_threshold}, Best Accuracy: {best_accuracy}")


def findthreshold(r):
    # Iterate over possible thresholds to find the best one
    best_threshold = None
    best_accuracy = 0
    for item in r:
        threshold = item["score_similarity"]
        print(threshold)
        accuracy = calculate_accuracy(threshold, r)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold


    print(f"Best Threshold: {best_threshold}, Best Accuracy: {best_accuracy}")
