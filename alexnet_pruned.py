'''
#Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
import tensorflow
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os
import numpy as np
from math import ceil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from math import ceil
batch_size = 128
num_classes = 10
epochs = 200
data_augmentation = True

config = tensorflow.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tensorflow.Session(config=config)
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'pruned_alexnet.h5'

f = open('final_results_pruned_alexnet.txt','w')

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

end_step = np.ceil(1.0 * 50000/128).astype(np.int32) * epochs
pruning_params = {
    'pruning_schedule' : sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                 final_sparsity = 0.90,
                                                 begin_step = 0,
                                                 end_step = end_step, frequency = 100 )

}


    
model = tensorflow.keras.Sequential([
    
# 1st Convolutional Layer
sparsity.prune_low_magnitude((tensorflow.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:], strides=(1, 1))), **pruning_params),
tensorflow.keras.layers.MaxPooling2D((2,2),(2,2), padding = 'valid'),   

# 2nd Convolutional Layer
sparsity.prune_low_magnitude((tensorflow.keras.layers.Conv2D(192, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:], strides=(1, 1))), **pruning_params),
tensorflow.keras.layers.MaxPooling2D((2,2),(2,2), padding = 'valid'),   

# 3rd Convolutional Layer
sparsity.prune_low_magnitude((tensorflow.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:], strides=(1, 1))), **pruning_params),

# 4th Convolutional Layer
sparsity.prune_low_magnitude((tensorflow.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:], strides=(1, 1))), **pruning_params),

# 5th Convolutional Layer
sparsity.prune_low_magnitude((tensorflow.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:], strides=(1, 1))), **pruning_params),
tensorflow.keras.layers.MaxPooling2D((2,2),(2,2), padding = 'valid'),   

# Passing it to a Fully Connected layer
tensorflow.keras.layers.Flatten(),

# 1st Fully Connected Layer
tensorflow.keras.layers.Dropout(0.4),
sparsity.prune_low_magnitude((tensorflow.keras.layers.Dense(2048, activation='relu')), **pruning_params),


# 2nd Fully Connected Layer
tensorflow.keras.layers.Dropout(0.4),
sparsity.prune_low_magnitude((tensorflow.keras.layers.Dense(2048, activation='relu')), **pruning_params),                             

# 3rd Fully Connected Layer


# Output Layer
sparsity.prune_low_magnitude((tensorflow.keras.layers.Dense(10, activation='softmax')), **pruning_params)

])

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-1
    if epoch > 180:
        lr *= 0.25e-3
    elif epoch > 160:
        lr *= 0.5e-3
    elif epoch > 135:
        lr *= 1e-3
    elif epoch > 90:
        lr *= 1e-2
    print('Learning rate: ', lr)
    return lr

opt = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule(0))
# Let's train the model using RMSprop
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

class PredictionCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        y_pred = model.predict(x_test)
        y_pred_bool= np.argmax(y_pred, axis=1)
        y_test_bool = np.argmax(y_test, axis=1)

        a = precision_score(y_test_bool, y_pred_bool, average="macro")
        f.write(str(a)+" ,")

callbacks = [ 
    pruning_callbacks.UpdatePruningStep(), pruning_callbacks.PruningSummaries(log_dir="saved_models"), PredictionCallback()]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test), callbacks = callbacks)"""

steps_per_epoch = ceil(50000/32)
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test), callbacks = callbacks,
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test), callbacks= callbacks,
                        workers=1)
final_model = sparsity.strip_pruning(model)
final_model.summary()

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
keras.models.save_model(final_model, model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])

f.write('\n')
f.write('val_accuracy : '+str(model.history.history['val_acc'])+'\n'+'accuracy : '+str(model.history.history['acc']))


plt.figure()
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["acc","val_acc"])
plt.savefig('accuracy_pruned_alexnet.png')
plt.show()



