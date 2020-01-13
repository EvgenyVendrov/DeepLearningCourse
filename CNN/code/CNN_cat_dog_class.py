from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

import numpy as np
from PIL import Image
import glob
import random


###
###!!!!!DOG=>0\\\\\CAT=>1!!!!!!#############
###

###functions###
def read_images(path_for_train_folder):
    all_data_tuples = []
    for filename in glob.glob(path_for_train_folder):
        im = Image.open(filename)
        np_array = (np.reshape(im, ((200 * 200) * 3)))
        if "dog" in filename:
            all_data_tuples.append((np_array, 0))
        else:
            all_data_tuples.append((np_array, 1))
    print(all_data_tuples)
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
    (data_x, data_y) = prepare_data(r"C:\Users\evgen\Desktop\data_set_for_3rdProj\train\*.jpg")
    print(data_x.shape)
    print(data_y.shape)
    ###notice that this is just an example for now
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))


if __name__ == '__main__':
    main()
