# DOGS=>class 0
# CATS=>class 1
from PIL import Image
import tensorflow as tf
import numpy as np
import glob
import random
import os
import time

###used to measure time learning took
start = time.time()

###os definitions for runing tensorflow faster###
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


###functions###
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


def shuffle_images(tuple_of_both_images_sets):
    train_set_data_tuples = []

    for image in tuple_of_both_images_sets[0]:
        train_set_data_tuples.append((image, 1))

    for image in tuple_of_both_images_sets[1]:
        train_set_data_tuples.append((image, 0))

    random.shuffle(train_set_data_tuples)
    random.shuffle(train_set_data_tuples)
    return train_set_data_tuples


def suffle_wo_reading(tuple_of_both_images_sets):
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


def prepare_data(path_for_cat_folder, path_for_dog_folder):
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


###main###
def main():
    # PLEASE NOTICE that again - this path is on MY computer - should be changed
    path_for_cat_folder = r'C:\Users\evgen\Desktop\proj1_DL\dataset\trainset\cat\*.jpg'
    path_for_dog_folder = r'C:\Users\evgen\Desktop\proj1_DL\dataset\trainset\dog\*.jpg'
    path_to_save_model = r"C:\Users\evgen\Desktop\proj1_DL\dataset\models\LR\MLP\250k\model.ckpt"
    # var to represent num of epochs
    num_of_epochs = 250000
    # hyper-parameter, learning rate
    learning_rate = 0.0001
    # our images are outputted as a-(100X100) vector so this is number of features - every pixels is a feature
    features = 2500

    # Hidden layers size#
    hidden_layer_node1 = 25
    hidden_layer_node2 = 10

    # declaring epsilon for escaping of numerical problems in loss function
    eps = 1e-12
    # creating a matrix [mX10,000] for x's from train data set
    x = tf.compat.v1.placeholder(tf.float32, [None, features], name="xs")
    # creating a vertical vector [mX1] for y's from train data set
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name="y_s")

    # Hidden layer 1#
    W1 = tf.Variable(tf.random.truncated_normal([features, hidden_layer_node1], stddev=0.3), name="Ws1")
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_node1]), name="bs1")
    z1 = tf.add(tf.matmul(x, W1), b1, name="z1_model_func")
    a1 = tf.nn.relu(z1, name="reLU1")

    # Hidden layer #2
    W2 = tf.Variable(tf.random.truncated_normal([hidden_layer_node1, hidden_layer_node2], stddev=0.3), name="Ws2")
    b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_node2]), name="bs2")
    z2 = tf.add(tf.matmul(a1, W2), b2, name="z2_model_func")
    a2 = tf.nn.relu(z2, name="reLU2")

    # logits layer
    # vertical vector [10,000X1] for weights on every feature
    W = tf.Variable(tf.random.truncated_normal([hidden_layer_node2, 1], stddev=0.3), name="Ws")
    # scalar bias starting from
    b = tf.Variable(tf.fill([1], 0.1), name="bs")
    # our model function
    y = tf.nn.sigmoid(tf.matmul(a2, W) + b, name="model_func")
    # our loss function we'll try to minimize
    sigmoid_loss = -(y_ * tf.math.log(y + eps) + (1 - y_) * tf.math.log(1 - y + eps))
    loss = tf.reduce_mean(sigmoid_loss, name="loss_func")
    # update algorithm
    update = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    # preparing data from given path for the learning
    tuple_of_both_images_sets = read_images(path_for_cat_folder, path_for_dog_folder)
    trainset_data_tupels = suffle_wo_reading(tuple_of_both_images_sets)
    data_x = np.asarray(trainset_data_tupels[0])
    data_y = np.asarray(trainset_data_tupels[1])
    # declaring our TF saver
    saver = tf.compat.v1.train.Saver()
    # declaring our TF session
    sess = tf.compat.v1.Session()
    # init the TF vars
    sess.run(tf.compat.v1.global_variables_initializer())
    # printing all learning parameters before first learning iteration
    print('Iteration:', 0, ' W1:', sess.run(W1), ' b1:', sess.run(b1), '\n' + ('#' * 50))
    print('Iteration:', 0, ' W2:', sess.run(W2), ' 2b:', sess.run(b2), '\n' + ('#' * 50))
    print('Iteration:', 0, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
          loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}), '\n' + ('#' * 50))
    # creating a var for printing how much we improved every thousand epochs
    last_loss_val = loss.eval(session=sess, feed_dict={x: data_x, y_: data_y})
    first_loss = last_loss_val

    # the learning itself
    for i in range(0, num_of_epochs):
        sess.run(update, feed_dict={x: data_x, y_: data_y})  # BGD
        if i % 1 == 0:
            print('Iteration:', 0, ' W1:', sess.run(W1), ' b1:', sess.run(b1), '\n' + ('#' * 50))
            print('Iteration:', 0, ' W2:', sess.run(W2), ' 2b:', sess.run(b2), '\n' + ('#' * 50))
            print('Iteration:', 0, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
                  loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}), '\n' + ('#' * 50))
            # if i == 0:
            #     print('Iteration:', (i + 1), ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
            #           loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}), 'improvement of loss:',
            #           last_loss_val - loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}), '\n' + ('#' * 50))
            # else:
            #     print('Iteration:', i, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
            #           loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}), 'improvement of loss:',
            #           last_loss_val - loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}), '\n' + ('#' * 50))
        last_loss_val = loss.eval(session=sess, feed_dict={x: data_x, y_: data_y})
        trainset_data_tupels = suffle_wo_reading(tuple_of_both_images_sets)
        data_x = np.asarray(trainset_data_tupels[0])
        data_y = np.asarray(trainset_data_tupels[1])

    # printing overall improvement after whole learning
    print('OVERALL loss improvement after ', num_of_epochs, ' iterations is==>',
          first_loss - loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}), '\n' + ('#' * 50))
    # saving the model for testing etc..
    saver.save(sess, path_to_save_model,
               global_step=100, write_meta_graph='false')
    #
    end = time.time()
    # printing to indicate that we finished the learning and saved the model in chosen path
    print('done learning and saved model to path =>', path_to_save_model, '\noverall run took =>', (end - start),
          ' seconds')


if __name__ == '__main__':
    main()
