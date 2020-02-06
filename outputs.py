from tensorflow.keras.models import load_model
from keras.datasets import cifar10
import tensorflow
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
# Convert class vectors to binary class matrices.
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
x_test = x_test.astype('float32')
x_test /= 255



model = load_model('saved_models/simplenetslim/optimizer_sgd_custom/simplenetslim.h5')

y_pred = model.predict(x_test)
#x_test_bool = np.argmax(x_test, axis=1)
y_pred_bool= np.argmax(y_pred, axis=1)
y_test_bool = np.argmax(y_test, axis=1)

print(confusion_matrix(y_test_bool,y_pred_bool))
#print(precision_score(y_test, y_pred, average="macro"))
#print(recall_score(y_test, y_pred, average="macro"))
#print(f1_score(y_test, y_pred, average= "macro"))

