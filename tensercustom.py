from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow и tf.keras
from tensorflow import keras

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt

shuffle_test_img = np.load('test_img.npy')  # loading
shuffle_test_label = np.load('test_label.npy')

class_names = ['Yes', 'No']

model = keras.models.load_model('tensorblack.h5')

test_lose, test_acc = model.evaluate(shuffle_test_img, shuffle_test_label)
print("\nТочность на проверочных данных: ", test_acc)
predictions = model.predict(shuffle_test_img)
print(predictions[0])
print(np.argmax(predictions[0]))
k = 0
plt.figure(figsize=(10, 10))
for i in range(2175, 2200):
    plt.subplot(5, 5, k + 1)
    k = k + 1
    plt.xticks([])
    plt.yticks([])
    plt.imshow(shuffle_test_img[i], cmap=plt.cm.binary)
    plt.grid(False)
    plt.xlabel('c:' + class_names[int(np.argmax(predictions[i]))] + '/n:' + class_names[shuffle_test_label[i]])
plt.show()
