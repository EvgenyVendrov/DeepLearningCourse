import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import random


###functions###
def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))


def return_class(p):
    return 'CAT' if p >= 0.5 else 'DOG'


def prepare_data(path_for_dog_folder, path_for_cat_folder):
    test_dog_images = []
    test_cat_images = []
    for image in glob.glob(path_for_dog_folder):
        im = Image.open(image)
        test_dog_images.append(np.reshape(im, (50 * 50)) / 255.)
    for image in glob.glob(path_for_cat_folder):
        im = Image.open(image)
        test_cat_images.append(np.reshape(im, (50 * 50)) / 255.)
    return (test_dog_images, test_cat_images)


def read_images(path_for_cat_folder, path_for_dog_folder):
    cat_folder = path_for_cat_folder
    dog_folder = path_for_dog_folder
    cat_image_list = []
    dog_image_list = []

    for filename in glob.glob(cat_folder):
        im = Image.open(filename)
        cat_image_list.append(np.reshape(im, (50 * 50)) / 255.)

    for filename in glob.glob(dog_folder):
        im = Image.open(filename)
        dog_image_list.append(np.reshape(im, (50 * 50)) / 255.)
    return (cat_image_list, dog_image_list)


def prepare_data2(path_for_cat_folder, path_for_dog_folder):
    tuple_of_both_images_sets = read_images(path_for_cat_folder, path_for_dog_folder)
    shuffled_tupels_of_all_data = shuffle_images(tuple_of_both_images_sets)
    train_data_x = []
    train_data_y = []
    for tuple_of_image_class in shuffled_tupels_of_all_data:
        train_data_x.append(tuple_of_image_class[0])

    for tuple_of_image_class in shuffled_tupels_of_all_data:
        train_data_y.append([tuple_of_image_class[1]])

    data_x = np.asarray(train_data_x)
    data_y = np.asarray(train_data_y)
    return (data_x, data_y)


def shuffle_images(tuple_of_both_images_sets):
    train_set_data_tuples = []

    for image in tuple_of_both_images_sets[0]:
        train_set_data_tuples.append((image, 1))

    for image in tuple_of_both_images_sets[1]:
        train_set_data_tuples.append((image, 0))

    random.shuffle(train_set_data_tuples)
    random.shuffle(train_set_data_tuples)
    return train_set_data_tuples


def load_model(path_for_model, name_for_model):
    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(
        path_for_model + name_for_model)
    saver.restore(sess,
                  tf.train.latest_checkpoint(path_for_model))

    graph = tf.get_default_graph()
    W = graph.get_tensor_by_name("Ws:0")
    b = graph.get_tensor_by_name("bs:0")
    x = graph.get_tensor_by_name("xs:0")
    y_ = graph.get_tensor_by_name("y_s:0")
    return (sess, W, b, x, y_)


###main###
def main():
    path_for_model = r'C:\Users\evgen\Desktop\proj1_DL\dataset\models\LR\MLP\250k'
    name_for_model = r'\model.ckpt-100.meta'
    path_for_dog_test = r'C:\Users\evgen\Desktop\proj1_DL\dataset\trainset\dog\*.jpg'
    path_for_cat_test = r'C:\Users\evgen\Desktop\proj1_DL\dataset\trainset\cat\*.jpg'
    tuples_of_all_data = prepare_data(path_for_dog_test, path_for_cat_test)

    pred_for_dog = []
    pred_for_cat = []

    (sess, W, b, x, y_) = load_model(path_for_model, name_for_model)

    for dog_image in tuples_of_all_data[0]:
        pred_for_dog.append(return_class(logistic_fun(np.matmul(dog_image, sess.run(W)) + sess.run(b))))

    num_of_dogs_recognized = 0
    index = 0
    for _class in pred_for_dog:
        if _class == 'DOG':
            num_of_dogs_recognized += 1

    for cat_image in tuples_of_all_data[1]:
        pred_for_cat.append(return_class(logistic_fun(np.matmul(cat_image, sess.run(W)) + sess.run(b))))

    num_of_cats_recognized = 0
    for _class in pred_for_cat:
        if _class == 'CAT':
            num_of_cats_recognized += 1
    (data_x, data_y) = prepare_data2(path_for_cat_test, path_for_dog_test)
    graph = tf.get_default_graph()
    loss = graph.get_tensor_by_name("loss_func:0")
    print(loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}))
    print('                         ***actual class***\n                            DOG:           CAT:')
    print('***Predicted class***')
    print('                      dog   ', num_of_dogs_recognized, '             ',
          len(pred_for_cat) - num_of_cats_recognized)
    print('                      cat   ', len(pred_for_dog) - num_of_dogs_recognized, '             ',
          num_of_cats_recognized)

    accuracy = ((num_of_cats_recognized + num_of_dogs_recognized) / (len(pred_for_dog) + len(pred_for_cat)))
    precision = num_of_dogs_recognized / (num_of_dogs_recognized + (len(pred_for_dog) - num_of_dogs_recognized))
    recall = num_of_dogs_recognized / (num_of_dogs_recognized + (len(pred_for_cat) - num_of_cats_recognized))

    print('model tasted from path: ', path_for_model)
    print('accuracy: %.3f' % (accuracy))
    print('precision: %.3f ' % (precision))
    print('recall: %.3f ' % (recall))
    print('f-measure: %.3f ' % (2 * ((precision * recall) / (precision + recall))))


if __name__ == '__main__':
    main()
