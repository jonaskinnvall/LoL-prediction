# Lib imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)

# Module imports
from ANN import evaluate, predict


def testNN(X, y):
    # Let user choose if they want to evaluate model
    # or use it to predict winner
    print("Do you want to evaluate model or predict winner?")
    response = None
    while response not in {"e", "p"}:
        response = input("Please enter 'e' or 'p': ")

    # Call evaluate function
    if response == 'e':
        evaluation = evaluate(X, y)
        # print('TEST LOSS, TEST ACC:', evaluation[0])
        print('TEST LOSS: %.3f, TEST ACC: %.3f%%' %
              (evaluation[0], evaluation[1]*100))

    # Call prediction function
    elif response == 'p':
        # Predict winner
        preds = predict(X)
        preds = np.round(preds)

        # Create and print confusion matrix and
        # classification report for predictions
        conf_mat = confusion_matrix(y, preds)
        class_rep = classification_report(y, preds)
        fpr, tpr, thresholds = roc_curve(y, preds)
        auc_value = auc(fpr, tpr)
        auc_score = roc_auc_score(y, preds)

        print('\n Conf Mat \n =============== \n', conf_mat)
        print('\n Class Rep \n =============== \n', class_rep)
        print('ROC AUC: %f' % auc_score)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_value)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
