#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import numpy as np

def get_TPR_FPR(predictions, targets, threshold):
    binary_predictions = np.greater(predictions, threshold)
    #fig, ax = plt.subplots()
    #plt.plot([1, 2, 3],[1, 1, 1])
    #plt.show()
    #plt.imshow(binary_predictions)
    #ax.plot(targets,'r')
    true_positives = np.sum(np.bitwise_and(binary_predictions, targets))
    false_positives = np.sum(binary_predictions) - true_positives
    # get negatives
    binary_predictions = 1 - binary_predictions
    targets = 1 - targets
    np.sum(np.bitwise_and(binary_predictions, targets))
    true_negatives = np.sum(np.bitwise_and(binary_predictions, targets))
    false_negatives = np.sum(binary_predictions) - true_negatives
    #false_negatives = np.sum(targets) - true_positives
    #negatives = binary_predictions.size - np.sum(binary_predictions)
    #true_negatives = negatives - false_negatives

    TPR = true_positives / (false_negatives + true_positives)
    FPR = false_positives / (false_positives + true_negatives)
    return TPR,FPR

def get_accuracy_precision_recall(predictions, targets, threshold=0.5):
    binary_predictions = np.greater(predictions, threshold)
    equal = np.equal(binary_predictions, targets)
    true_positives = np.sum(np.bitwise_and(binary_predictions, targets))

    accuracy = 100 * np.mean(equal)

    if np.sum(binary_predictions) == 0:
        precision = 100.0
    else:
        precision = 100 * true_positives / np.sum(binary_predictions)

    if np.sum(targets) == 0:
        recall = 100.0
    else:
        recall = 100 * true_positives / np.sum(targets)

    return accuracy, precision, recall
