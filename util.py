"""
    Contains Utility Methods
"""


def normalize_predictions(predictions, cutoff=0.5):
    return [0 if x < cutoff else 1 for x in predictions]


def get_precision(true_pos, false_pos):
    true_pos_plus_false_pos = [x + y for x, y in zip(true_pos, false_pos)]
    precision = [0 if y == 0 else x / y for x, y in zip(true_pos, true_pos_plus_false_pos)]
    return precision


def get_recall(true_pos, false_neg):
    true_pos_plus_false_neg = [x + y for x, y in zip(true_pos, false_neg)]
    recall = [0 if y == 0 else x / y for x, y in zip(true_pos, true_pos_plus_false_neg)]
    return recall


def get_f1_score(precision_list, recall_list):
    f1 = lambda precision, recall: 2 * precision * recall / (precision + recall)
    return [0 if y == 0 or x == 0 else f1(x, y) for x, y in zip(precision_list, recall_list)]


def get_confusion_matrix(labels, predictions):
    true_positives = [0] * 8
    true_negatives = [0] * 8
    false_positives = [0] * 8
    false_negatives = [0] * 8

    for label, prediction in zip(labels, predictions):
        for i in range(8):
            if label[i] == prediction[i]:
                if prediction[i] == 1:
                    true_positives[i] += 1
                else:
                    true_negatives[i] += 1
            else:
                if prediction[i] == 1:
                    false_positives[i] += 1
                else:
                    false_negatives[i] += 1

    print("True Pos:", true_positives)
    print("True Neg:", true_negatives)
    print("False Pos:", false_positives)
    print("False Neg:", false_negatives)

    return true_positives, true_negatives, false_positives, false_negatives


def print_summary(labels, predictions):
    classes = ("Report", "Device", "Delivery", "Progress", "Becoming_Member", "Attempt_Action", "Activity", "Other")

    true_pos, true_neg, false_pos, false_neg = get_confusion_matrix(labels, predictions)

    precision = get_precision(true_pos, false_pos)

    recall = get_recall(true_pos, false_neg)

    f1_score = get_f1_score(precision, recall)

    print("\nSummary:")

    for i in range(8):
        print("Class -", classes[i])
        print("-----------------------")
        print("Precision:", precision[i])
        print("Recall:", recall[i])
        print("F1 Score:", f1_score[i])
        print("-----------------------\n")
