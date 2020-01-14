import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot
import numpy as np
from PIL import Image
import glob
import random
import sys
import json



###
###!!!!!DOG=>0\\\\\CAT=>1!!!!!!#############
###

###functions###
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


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


#
#
# def train_loop(features, labels):
#     # Define the GradientTape context
#     with tf.GradientTape() as tape:
#         # Get the probabilities
#         predictions = model(features)
#         # Calculate the loss
#         loss = loss_func(labels, predictions)
#     # Get the gradients
#     gradients = tape.gradient(loss, model.trainable_variables)
#     # Update the weights
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss


def main():
    num_of_epochs = 1
    # preparing the data
    (data_x, data_y) = prepare_data(r"C:\Users\evgen\Desktop\data_set_for_3rdProj\train\*.jpg")
    (val_x, val_y) = prepare_data(r"C:\Users\evgen\Desktop\data_set_for_3rdProj\validation\*.jpg")
    (test_x, test_y) = prepare_data(r"C:\Users\evgen\Desktop\data_set_for_3rdProj\test\*.jpg")
    print(data_x.shape)
    print(data_y.shape)
    ##notice that this is just an example for now
    ##building the model
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
               input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    binary_cross_entropy_loss_func = tf.keras.losses.BinaryCrossentropy()
    ADAM_optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=ADAM_optimizer, loss=binary_cross_entropy_loss_func, metrics=['accuracy'])
    history = model.fit(data_x, data_y,
                        batch_size=64,
                        epochs=num_of_epochs,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(val_x, val_y))
    print('\nhistory dict:', history.history)

    #model.save(r"C:\Users\evgen\Desktop\models\model.h5", True / False, False)
    json_str = model.to_json()
    with open(r'C:\Users\evgen\Desktop\models\saved_model.json', 'w') as outfile:
        json.dump(json.loads(json_str), outfile, indent=4)    # Save the json on a file
    model.save_weights(r"C:\Users\evgen\Desktop\models\weights.h5", save_format="h5")
    # model_json = model.to_json()
    # with open(r"C:\Users\evgen\Desktop\models\model.h5", "w") as json_file:
    #     json_file.write(model_json)
    #     # serialize weights to HDF5
    #     model.save_weights("model.h5")
    print("Saved model to disk")
    # summarize_diagnostics(history)
    #
    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(test_x, test_y, batch_size=128)
    print('test loss, test acc:', results)

    # # Generate predictions (probabilities -- the output of the last layer)
    # # on new data using `predict`
    # # print('\n# Generate predictions for 3 samples')
    # # predictions = model.predict(x_test[:3])
    # # print('predictions shape:', predictions.shape)
    ##SAVING - NOT NEEDED
    # # saver.save(sess, path_to_save_model,
    # #            global_step=100, write_meta_graph='false')
    # #


if __name__ == '__main__':
    main()
