import cv2
import tensorflow as tf
import Model
import matplotlib.pyplot as plt
import numpy as np


@tf.function
def load_data(show_images=False, batch_size=16):
    data = tf.data.Dataset.list_files("./data/*.jpg", shuffle=False) \
        .map(lambda x: tf.io.decode_jpeg(tf.io.read_file(x)) / 255)
    labels = tf.data.Dataset.list_files("./labels/*.jpg", shuffle=False) \
        .map(lambda x: tf.io.decode_jpeg(tf.io.read_file(x)) / 255)

    dataset = tf.data.Dataset.zip((data, labels))

    if show_images:
        a_list = [(i[0].numpy(), i[1].numpy()) for i in dataset.take(1)]
        cv2.imshow("Data", a_list[0][0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fig, axe = plt.subplots(nrows=1, ncols=2)
        axe[0].imshow(a_list[0][0])
        axe[0].set_title("Data")
        axe[1].imshow(a_list[0][1])
        axe[0].set_title("Ground Truth")
        plt.show()

    dataset = dataset.batch(batch_size)
    dataset = dataset.take(2)
    return dataset


epochs = 20
model_type = "convolutional"
model = Model.get_model(model_type)
training_dataset = load_data()
model.fit(training_dataset, epochs=epochs)

image = cv2.imread("./data/1.jpg") / 255
result = model.predict(np.expand_dims(image, axis=0))
result = np.reshape(result, (480, 640, 3)) * 255
cv2.imwrite(f"./result_{epochs}_epochs_{model_type}.jpg", result)
model.save(f"models/{model_type}_{epochs}_epochs.h5")

# https://www.tensorflow.org/tutorials/load_data/images?hl=en
