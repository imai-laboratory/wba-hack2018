import os
import cv2
import tensorflow as tf
import numpy as np

def load_image(file_path):
    module_dir, _ = os.path.split(os.path.realpath(__file__))
    absolute_path = os.path.join(module_dir, file_path)
    image = cv2.imread(absolute_path)
    # (h, w, c), uint8
    # Change BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(image, file_path):
    module_dir, _ = os.path.split(os.path.realpath(__file__))
    absolute_path = os.path.join(module_dir + "/../..", file_path)

    # Change RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(absolute_path, image)

def softmax(values, temp=0.1):
    values /= temp
    e_x = np.exp(values - np.max(values))
    return e_x / e_x.sum(axis=0)

# create varible list for saver while replacing top
# level namespace with specific name
def create_variable_saver(variables, from_name, to_name):
    var_list = {}
    for var in variables:
        key = var.name
        # -2 is for ':0'
        var_list[key.replace(from_name, to_name)[:-2]] = var
    return tf.train.Saver(var_list)
