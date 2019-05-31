import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, MaxPooling2D, Input, Flatten, Dropout 
from keras import optimizers
from keras.utils import to_categorical
import sys
import os


DATA_DIR = os.path.join(os.getcwd(),'data')
TRAIN_DIR = os.path.join(DATA_DIR,'seg_train')
VAL_DIR = os.path.join(DATA_DIR,'seg_pred')
TEST_DIR = os.path.join(DATA_DIR,'seg_test')

category = [ 'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street' ]

CLASS_NUM = 6
EPOCHS = 100

for cat in category:
    print("The total pics in {} folder are {}".format(cat, len(os.listdir(os.path.join(TRAIN_DIR, cat)))))


def convframe(path, category):
    train_data = []
    labels = []
    data_list = []
    images = []
    for cat in category:
        cat_path = os.path.join(path,cat)

        for data in os.listdir(cat_path):
            image = cv2.imread(cat_path + '/' + data)
            image = cv2.resize(image, (150,150))
            images.append(image)
            
            if cat == 'buildings':
                label = 0
            elif cat == 'forest':
                label = 1
            elif cat == 'glacier':
                label = 2
            elif cat == 'mountain':
                label = 3
            elif cat == 'sea':
                label = 4
            elif cat == 'street':
                label = 5
            labels.append(label)
    
    return images, labels

def net(images, labels, test_images, test_labels):
    data = Input(shape=(150,150,3))

#conv1
    conv1_1 = Conv2D(64, (3,3), padding='same')(data)
    bn1_1 = BatchNormalization()(conv1_1)
    act1_1 = Activation('relu')(bn1_1)

    conv1_2= Conv2D(64, (3,3), padding='same')(act1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    act1_2 = Activation('relu')(bn1_2)
    pool1 = MaxPooling2D((2,2), strides=2)(act1_2)
# 75x75x64

# conv2
    conv2_1 = Conv2D(128, (3,3), padding='same')(pool1)
    bn2_1 = BatchNormalization()(conv2_1)
    act2_1 = Activation('relu')(bn2_1)

    conv2_2 = Conv2D(128, (3,3), padding='same')(act2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    act2_2 = Activation('relu')(bn2_2)
    pool2 = MaxPooling2D((2,2), strides=2, padding='same')(act2_2)
# 38x38x128 

# conv3
    conv3_1 = Conv2D(256, (3,3), padding='same')(pool2)
    bn3_1 = BatchNormalization()(conv3_1)
    act3_1 = Activation('relu')(bn3_1)

    conv3_2 = Conv2D(256, (3,3), padding='same')(act3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    act3_2 = Activation('relu')(bn3_2)

    conv3_3 = Conv2D(256, (3,3), padding='same')(act3_2)
    bn3_3 = BatchNormalization()(conv3_3)
    act3_3 = Activation('relu')(bn3_3)
    pool3 = MaxPooling2D((2,2), strides=2)(act3_3)
# 19x19x256

# conv4
    conv4_1 = Conv2D(512, (3,3), padding='same')(pool3)
    bn4_1 = BatchNormalization()(conv4_1)
    act4_1 = Activation('relu')(bn4_1)

    conv4_2 = Conv2D(512, (3,3), padding='same')(act4_1)
    bn4_2= BatchNormalization()(conv4_2)
    act4_2 = Activation('relu')(bn4_2)

    conv4_3 = Conv2D(512, (3,3), padding='same')(act4_2)
    bn4_3 = BatchNormalization()(conv4_3)
    act4_3 = Activation('relu')(bn4_3)
    pool4 = MaxPooling2D((2,2), strides=2, padding='same')(act4_3)
# 10x10x512

# conv5
    conv5_1 = Conv2D(512, (3,3), padding='same')(pool4)
    bn5_1 = BatchNormalization()(conv5_1)
    act5_1 = Activation('relu')(bn5_1)

    conv5_2 = Conv2D(512, (3,3), padding='same')(act5_1)
    bn5_2= BatchNormalization()(conv5_2)
    act5_2 = Activation('relu')(bn5_2)

    conv5_3 = Conv2D(512, (3,3), padding='same')(act5_2)
    bn5_3 = BatchNormalization()(conv5_3)
    act5_3 = Activation('relu')(bn5_3)
    pool5 = MaxPooling2D((2,2), strides=2)(act5_3)
# 5x5x512

    flat = Flatten()(pool5)
    hidden1 = Dense(4096, activation='relu')(flat)
    drop1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(4096, activation='relu')(drop1)
    drop2 = Dropout(0.5)(hidden2)
    output = Dense(CLASS_NUM, activation='softmax')(drop2)
    model = Model(inputs=data, outputs=output)

    print(model.summary())
    
    opt = optimizers.SGD(0.001, 0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(images, to_categorical(labels, CLASS_NUM), batch_size=32, epochs=EPOCHS, validation_data=(test_images, to_categorical(test_labels, CLASS_NUM)), shuffle=True)
    model.save('vgg19.h5')
    del model


images, labels = convframe(TRAIN_DIR, category)
test_images, test_labels = convframe(TEST_DIR, category)

labels = np.array(labels)
images = np.array(images)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

'''
for idx in range(len(train_data)):
    if images[idx].shape[0] is not 150 or images[idx].shape[1] is not 150:
        print(images[idx].shape)
        num +=1
'''

net(images, labels, test_images, test_labels)

