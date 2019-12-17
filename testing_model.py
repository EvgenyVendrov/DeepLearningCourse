import numpy as np
import tensorflow as tf
from PIL import Image
import glob


def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))


def return_class(p):
    return 'CAT' if p >= 0.5 else 'DOG'


sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(
    r'C:\Users\evgen\Desktop\ML\V2\saved_model\model.ckpt-100.meta')
saver.restore(sess,
              tf.train.latest_checkpoint(r'C:\Users\evgen\Desktop\ML\V2\saved_model'))

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("xs:0")
y_ = graph.get_tensor_by_name("y_s:0")
W = graph.get_tensor_by_name("Ws:0")
b = graph.get_tensor_by_name("bs:0")
y = graph.get_tensor_by_name("model_func:0")
loss = graph.get_tensor_by_name("loss_func:0")
#
# test_cat = r'C:\Users\evgen\Desktop\ML\V2\test_set\cats\0.jpg'
# test_dog = r'C:\Users\evgen\Desktop\ML\V2\test_set\dogs\34.jpg'
# im_cat = Image.open(test_cat)
# im_dog = Image.open(test_dog)
# test_data = []
# test_data.append(np.reshape(im_cat, (100 * 100)))
# test_data.append(np.reshape(im_dog, (100 * 100)))
# feed_dict = {x: test_data}
# pred_for_cat = return_class(logistic_fun(np.matmul(test_data[0], sess.run(W)) + sess.run(b)))
# pred_for_dog = return_class(logistic_fun(np.matmul(test_data[1], sess.run(W)) + sess.run(b)))
# print(np.matmul(test_data[1], sess.run(W)) + sess.run(b))
# print(np.matmul(test_data[0], sess.run(W)) + sess.run(b))
# print('after func cat=>', logistic_fun(np.matmul(test_data[0], sess.run(W)) + sess.run(b)))
# print('after func dog=>', logistic_fun(np.matmul(test_data[1], sess.run(W)) + sess.run(b)))
# print('should be cat:', pred_for_cat)
# print('should be dog:', pred_for_dog)
folder_dog = r'C:\Users\evgen\Desktop\ML\V2\test_set\dogs\*.jpg'
all_dog = []
pred_for_dog = []
for filename in glob.glob(folder_dog):
    im = Image.open(filename)
    all_dog.append(np.reshape(im, (100 * 100)) / 255.)

for image in all_dog:
    pred_for_dog.append(return_class(logistic_fun(np.matmul(image, sess.run(W)) + sess.run(b))))
num_of_dogs = 0
for clas in pred_for_dog:
    if clas == 'DOG':
        num_of_dogs += 1

print(len(pred_for_dog) - num_of_dogs)
print(1.0 * ((len(pred_for_dog) - num_of_dogs) / len(pred_for_dog)))
print('overall image tested: ',
      len(pred_for_dog), '\nimages classified as dog: ', num_of_dogs, '\nimages we got wrong: ',
      len(pred_for_dog) - num_of_dogs)
