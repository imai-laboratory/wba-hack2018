import numpy as np
import math
import cv2

from sklearn.model_selection import train_test_split


def read_dataset(path, image_size, data_size=1e5, batch_size=64, num_epochs=100):
    images = np.load(path)[:data_size]
    train_images, test_images = train_test_split(
        images, test_size=0.01, shuffle=True)

    train_size = train_images.shape[0]
    test_size = test_images.shape[0]

    num_train_iterations = math.ceil(train_size / batch_size)
    num_test_iterations = math.ceil(test_size / batch_size)

    def get_next(callback=None):
        for epoch in range(num_epochs):
            for i in range(num_train_iterations):
                index = i * batch_size
                if index + batch_size > train_size:
                    images = train_images[index:]
                else:
                    images = train_images[index:index + batch_size]
                resized_images = []
                for image in images:
                    resized_images.append(cv2.resize(image, image_size))
                yield np.array(resized_images)
            if callback is not None:
                callback(epoch)

    def get_test():
        while True:
            for i in range(num_test_iterations):
                index = i * batch_size
                if index + batch_size > test_size:
                    images = test_images[index:]
                else:
                    images = test_images[index:index + batch_size]
                resized_images = []
                for image in images:
                    resized_images.append(cv2.resize(image, image_size))
                yield np.array(resized_images)

    return get_next, get_test
