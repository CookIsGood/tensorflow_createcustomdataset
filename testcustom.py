from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Создание тренировачных входных данных
DATADIR = "A:\python\hatetensorflow\img"
CATEGORIES = ["True", "False"]
IMG_SIZE = 600
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
            except Exception as e:
                pass


create_training_data()

training_img = []
training_label = []
for features, label in training_data:
    training_img.append(features)
    training_label.append(label)
np.save('train_img.npy', training_img)  # saving
training_img = np.load('train_img.npy')  # loading
np.save('train_label.npy', training_label)
training_label = np.load('train_label.npy')
print("Размер и количество тренировочных картинок", training_img.shape)
print("Количество тренировочных меток", len(training_label))
# Создание тестовых входных данных
TESTDATADIR = "A:\python\hatetensorflow\img1"
TESTCATEGORIES = ["True", "False"]
test_data = []


def create_test_data():
    for category in TESTCATEGORIES:
        path = os.path.join(TESTDATADIR, category)
        class_num = TESTCATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])
            except Exception as e:
                pass


create_test_data()

test_img = []
test_label = []
for features, label in test_data:
    test_img.append(features)
    test_label.append(label)
np.save('test_img.npy', test_img)  # saving
test_img = np.load('test_img.npy')  # loading
np.save('test_label.npy', test_label)
test_label = np.load('test_label.npy')
# training_img = training_img / 255
# test_img = test_img / 255

print("Размер и количество тестовых картинок", test_img.shape)
print("Количество тестовых меток", len(test_label))

model3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model3.compile(optimizer='Adadelta',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model3.fit(training_img, training_label, epochs=10)
test_lose, test_acc = model3.evaluate(test_img, test_label)
print("\nТочность на проверочных данных: ", test_acc)
model3.save('tensorblack.h5')
print('Модель успешно сохранена')
