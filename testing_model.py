import numpy as np
import tensorflow as tf
from PIL import Image
import glob


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
        test_dog_images.append(np.reshape(im, (100 * 100)) / 255.)
    for image in glob.glob(path_for_cat_folder):
        im = Image.open(image)
        test_cat_images.append(np.reshape(im, (100 * 100)) / 255.)
    return (test_dog_images, test_cat_images)


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
    return (sess, W, b)


###main###
def main():
    path_for_model = r'C:\Users\evgen\Desktop\ML\V2\saved_model'
    name_for_model = r'\model.ckpt-100.meta'
    path_for_dog_test = r'C:\Users\evgen\Desktop\ML\V2\test_set\dogs\*.jpg'
    path_for_cat_test = r'C:\Users\evgen\Desktop\ML\V2\test_set\cats\*.jpg'
    tuples_of_all_data = prepare_data(path_for_dog_test, path_for_cat_test)

    pred_for_dog = []
    pred_for_cat = []

    (sess, W, b) = load_model(path_for_model, name_for_model)

    for dog_image in tuples_of_all_data[0]:
        pred_for_dog.append(return_class(logistic_fun(np.matmul(dog_image, sess.run(W)) + sess.run(b))))

    num_of_dogs_recognized = 0
    index = 0
    for _class in pred_for_dog:
        if _class == 'DOG':
            num_of_dogs_recognized += 1

    for cat_image in tuples_of_all_data[1]:
        pred_for_cat.append(return_class(logistic_fun(np.matmul(cat_image, sess.run(W)) + sess.run(b))))

    #     index = 0
    num_of_cats_recognized = 0
    for _class in pred_for_cat:
        if _class == 'CAT':
            num_of_cats_recognized += 1


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
