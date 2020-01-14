import random
import glob
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import keras
import json
import tensorflow as tf



def read_images(path_for_train_folder):
    all_data_tuples = []
    for filename in glob.glob(path_for_train_folder):
        im = Image.open(filename)
        np_array = (np.reshape(im, ((200, 200, 3))))
        if "dog" in filename:
            all_data_tuples.append((np_array, 0))
        else:
            all_data_tuples.append((np_array, 1))
    return all_data_tuples


def shuffle_images(tuple_of_both_images_sets):
    shuff_tuples = random.shuffle(tuple_of_both_images_sets)
    return shuff_tuples


def prepare_data(path_for_train_folder):
    all_train_set_tuples = read_images(path_for_train_folder)
    shuffled_tuples_of_all_data = random.sample(all_train_set_tuples, len(all_train_set_tuples))
    train_data_x = []
    train_data_y = []

    for tuple_of_image_class in shuffled_tuples_of_all_data:
        train_data_x.append(tuple_of_image_class[0])
    for tuple_of_image_class in shuffled_tuples_of_all_data:
        train_data_y.append([tuple_of_image_class[1]])

    data_x = np.asarray(train_data_x)
    data_y = np.asarray(train_data_y)
    return (data_x, data_y)
def main():

    with open(r"C:\Users\evgen\Desktop\models\saved_model.json") as json_file:
        data = json.load(json_file)
    data = json.dumps(data)
    model = tf.keras.models.model_from_json(data)
    model.load_weights(r"C:\Users\evgen\Desktop\models\weights.h5")
    binary_cross_entropy_loss_func = tf.keras.losses.BinaryCrossentropy()
    ADAM_optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=ADAM_optimizer, loss=binary_cross_entropy_loss_func, metrics=['accuracy'])
    (test_x, test_y) = prepare_data(r"C:\Users\evgen\Desktop\data_set_for_3rdProj\test\*.jpg")
    results = model.evaluate(test_x, test_y, batch_size=128)
    print('test loss, test acc:', results)

if __name__ == '__main__':
    main()