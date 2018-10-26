""" Taken from: http://scikit-learn.org/stable/auto_examples/model_selection/get_confusion_metrics.html """
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, \
    balanced_accuracy_score

from config import settings


def get_confusion_metrics(y_test, y_pred, epoch_no=0,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix


    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, [0, 1], settings.class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    np.set_printoptions(precision=2)
    print(cm)
    # print(report)

    """ CALCULATE METRICS """
    if settings.num_classes == 2:
        average = 'binary'
    elif settings.num_classes > 2:
        average = 'micro'
    else:
        average = 'Only one class provided.'

    F1_score = round(f1_score(y_test, y_pred, average=average), 3)
    MCC = round(matthews_corrcoef(y_test, y_pred), 3)
    informedness = round(balanced_accuracy_score(y_test, y_pred, adjusted=True), 3)
    # http: // scikit - learn.org / stable / modules / model_evaluation.html

    if epoch_no == settings.epochs:
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(settings.class_names))
        plt.xticks(tick_marks, settings.class_names, rotation=45)
        plt.yticks(tick_marks, settings.class_names)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.grid(b=False)

        plt.savefig(settings.output_dir + '/figures/confusion_{}_epoch{}.png'.format(settings.iteration_name, epoch_no))
        plt.close()

    return informedness, F1_score, MCC



