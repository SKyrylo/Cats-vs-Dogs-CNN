import numpy as np
import matplotlib
# import matplotlib.pyplot as plt
import os
import cv2
# import random
import pickle
import keras
from keras import Sequential, layers
from keras import callbacks
import time

matplotlib.use('TkAgg')

DATADIR = 'C:\\Users\\Kyrylo\\Desktop\\NeuralNets\\Cats-vs-Dogs'
CATEGORIES = ['Dogs', 'Cats']
IMG_SIZE = 50
training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except ValueError:
                pass


# create_training_data()
# random.shuffle(training_data)
# print(len(training_data))
# for sample in training_data[:10]:
#     print(sample[1])
#
# X = []
# y = []
#
# for features, label in training_data:
#     X.append(features)
#     y.append(label)
#
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#
# pickle_out = open('X.pickle', 'wb')
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open('y.pickle', 'wb')
# pickle.dump(y, pickle_out)
# pickle_out.close()


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

y = np.array(y)
X = X/255.0

dense_layers = [2]
layer_sizes = [128]
conv_layers = [3]

# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers:
#             NAME = f'{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}'
#             tensorboard = callbacks.TensorBoard(log_dir=f'logs/{NAME}')
#
#             model = Sequential()
#             model.add(layers.Conv2D(layer_size, (3, 3), input_shape=X.shape[1:], activation='relu'))
#             model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
#             for _ in range(conv_layer-1):
#                 model.add(layers.Conv2D(layer_size, (3, 3), activation='relu'))
#                 model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
#             model.add(layers.Flatten())
#
#             for _ in range(dense_layer):
#                 model.add(layers.Dense(512, activation='relu'))
#
#             model.add(layers.Dense(1, activation='sigmoid'))
#
#             model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#             model.fit(X, y, batch_size=32, validation_split=0.1, epochs=3, callbacks=[tensorboard])
#
#             model.save(f'models\\{NAME}.keras', save_format='keras')


model = keras.models.load_model('models\\3-conv-128-nodes-2-dense-1736011127.keras')


def prepare(filepath):
    img_size = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


prediction = model.predict([prepare('Cats/cat.1.jpg')])
prediction2 = model.predict([prepare('Dogs/dog.10.jpg')])
print(prediction, prediction2)
