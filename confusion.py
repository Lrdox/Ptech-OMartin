from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.models import load_model
from keras.datasets import cifar10
import tensorflow
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
# Convert class vectors to binary class matrices.
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
x_test = x_test.astype('float32')
x_test /= 255


model = load_model('saved_models/alexnet/sgd_custom/alexnet.h5')

y_pred = model.predict(x_test)
y_pred_bool= np.argmax(y_pred, axis=1)
y_test_bool = np.argmax(y_test, axis=1)

def plot_confusion_matrix(cm,class_,title='Confusion matrix alexnet',cmap=plt.cm.Blues):
    """
    This function plots a confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(class_))
    plt.xticks(tick_marks, class_, rotation=90)
    plt.yticks(tick_marks, class_)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig('confusion_alexnet.png')


# use scikit-learn to generate confusion matrix
cm = confusion_matrix(y_test_bool, y_pred_bool)
# plot matrix
plot_confusion_matrix(cm, ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])

#['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']