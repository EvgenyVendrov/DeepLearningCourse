from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import tensorflow as tf

batch_size = 64
IMG_HEIGHT = 200
IMG_WIDTH = 200
test_image_generator = ImageDataGenerator(rescale=1. / 255)
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=r"C:\Users\evgen\Desktop\DL_Git_forAllProj\CNN\data_for_examp",
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')

with open(r"C:\Users\evgen\Desktop\DL_Git_forAllProj\CNN\saved_models\saved_model_250ep_w_dropout_data_rich.json") as json_file:
    data = json.load(json_file)
    data = json.dumps(data)
    model = tf.keras.models.model_from_json(data)
model.load_weights(r"C:\Users\evgen\Desktop\DL_Git_forAllProj\CNN\saved_models\weights_250ep_w_dropout_data_rich.h5")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
results_test = model.evaluate_generator(test_data_gen)
print('test loss, test acc:, Precision:, Recall', results_test)
acc = results_test[1]
precision = results_test[2]
recall = results_test[3]
f_measure = 2 * (precision * recall) / (precision + recall)
print("acc: ", acc)
print("F-Measure: ", f_measure)
