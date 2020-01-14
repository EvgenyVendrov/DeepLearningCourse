# ####first arg => where to
# ####second arg=> from where
# ####third arg=> which postfix
#
#
# from PIL import Image
# import glob
# import os
# import sys
#
# postfix = sys.argv[5]
# path_for_dog_train = sys.argv[1] + r'\*.' + postfix
# path_for_cat_train = sys.argv[2] + r'\*.' + postfix
# path_for_dog_test = sys.argv[3] + r'\*.' + postfix
# path_for_cat_test = sys.argv[4] + r'\*.' + postfix
# folder_for_train = sys.argv[6]
# folder_for_test = sys.argv[7]
#
# num = 0
# for filename in glob.glob(path_for_dog_train):
#     im = Image.open(filename)
#     im = im.resize((200, 200), Image.ANTIALIAS)
#     new_path = os.path.join(folder_for_train, "dog" + str(num)) + '.' + postfix
#     im.save(new_path)
#     num += 1
# print("finished ", num, " images of dog_train")
#
# num = 0
# for filename in glob.glob(path_for_cat_train):
#     im = Image.open(filename)
#     im = im.resize((200, 200), Image.ANTIALIAS)
#     new_path = os.path.join(folder_for_train, "cat" + str(num)) + '.' + postfix
#     im.save(new_path)
#     num += 1
# print("finished ", num, " images of cat_train")
#
# num = 0
# for filename in glob.glob(path_for_dog_test):
#     im = Image.open(filename)
#     im = im.resize((200, 200), Image.ANTIALIAS)
#     new_path = os.path.join(folder_for_test, "dog" + str(num)) + '.' + postfix
#     im.save(new_path)
#     num += 1
# print("finished ", num, " images of dog_test")
#
# num = 0
# for filename in glob.glob(path_for_cat_test):
#     im = Image.open(filename)
#     im = im.resize((200, 200), Image.ANTIALIAS)
#     new_path = os.path.join(folder_for_test, "cat" + str(num)) + '.' + postfix
#     im.save(new_path)
#     num += 1
# print("finished ", num, " images of cat_test")
import tensorflow as tf
print(tf.version.VERSION)
