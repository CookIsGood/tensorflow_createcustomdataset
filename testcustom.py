from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
from tensorflow import keras

# Вспомогательные библиотеки
import numpy as np
import os
import cv2

# Создание тренировачных входных данных
DATADIR = "A:\python\createcustomtensorflow\img"
CATEGORIES = ["True", "False"]
IMG_SIZE = 300
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

c1 = np.c_[training_img.reshape(len(training_img), -1), training_label.reshape(len(training_label), -1)]
np.random.shuffle(c1)
shuffle_training_img = c1[:, :training_img.size//len(training_img)].reshape(training_img.shape)
shuffle_training_label = c1[:, training_img.size//len(training_img):].reshape(training_label.shape)
np.save('train_img.npy', shuffle_training_img)
np.save('train_label.npy', shuffle_training_label)


print("Размер и количество тренировочных картинок", shuffle_training_img.shape)
print("Количество тренировочных меток", len(shuffle_training_label))
# Создание тестовых входных данных
TESTDATADIR = "A:\python\createcustomtensorflow\img1"
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

c = np.c_[test_img.reshape(len(test_img), -1), test_label.reshape(len(test_label), -1)]
np.random.shuffle(c)
shuffle_test_img = c[:, :test_img.size//len(test_img)].reshape(test_img.shape)
shuffle_test_label = c[:, test_img.size//len(test_img):].reshape(test_label.shape)
np.save('test_img.npy', shuffle_test_img)
np.save('test_label.npy', shuffle_test_label)


print("Размер и количество тестовых картинок", shuffle_test_img.shape)
print("Количество тестовых меток", len(shuffle_test_label))
input_shape = (IMG_SIZE, IMG_SIZE)
model3 = keras.Sequential([
    keras.layers.Conv1D(3, 1, padding="same", activation="relu", input_shape=input_shape),
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model3.compile(optimizer='Adadelta',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model3.fit(shuffle_training_img, shuffle_training_label, epochs=10)
test_lose, test_acc = model3.evaluate(shuffle_test_img, shuffle_test_label)
print("\nТочность на проверочных данных: ", test_acc)
model3.save('tensorblack.h5')
print('Модель успешно сохранена')
