import numpy as np
from sklearn.metrics import confusion_matrix

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization

""" Parameters """
batch_size = 512
num_classes = 10
epoch = 1000

""" Get Data """
(train_data, train_label), (test_data, test_label) = cifar10.load_data()

""" One Hot Encoding """
train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)

""" Model Configuration """
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=train_data.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

""" Optimizer """
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-7)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

""" Normalize Data """
train_data = train_data / 255
test_data = test_data / 255


train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
for i in range(epoch):
    model.fit(train_data, train_label,
              batch_size=batch_size,
              epochs=int(epoch/epoch),
              validation_data=(test_data, test_label),
              shuffle=True)

    train_record = model.evaluate(train_data, train_label)
    train_loss_list.append(train_record[0])
    train_acc_list.append(train_record[1])

    val_record = model.evaluate(test_data, test_label)
    val_loss_list.append(val_record[0])
    val_acc_list.append(val_record[1])

print("\nAverage testing accuracy: {}%".format(round(sum(val_acc_list)/len(val_acc_list), 4) * 100))
print("Maximum testing accuracy: {}%".format(round(max(val_acc_list), 4) * 100))
print("Last epoch testing accuracy: {}%".format(round(val_acc_list[-1], 4) * 100))
