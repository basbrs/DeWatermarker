import os
import tensorflow as tf
import numpy as np

from tqdm import trange, tqdm
from datetime import datetime
from PIL import Image

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Model


INPUT_SHAPE = (480, 640, 3)
INITIAL_FILTER_SIZE = 64


def _build_autoencoder():
    encoder = Input(shape=INPUT_SHAPE)
    x = Conv2D(filters=INITIAL_FILTER_SIZE, kernel_size=(3, 3), activation="elu", padding="same")(encoder)
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = Conv2D(filters=INITIAL_FILTER_SIZE // 2, kernel_size=(3, 3), activation="elu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = Conv2D(filters=INITIAL_FILTER_SIZE // 4, kernel_size=(3, 3), activation="elu", padding="same")(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    decoder = Conv2D(filters=INITIAL_FILTER_SIZE // 4, kernel_size=(3, 3), activation="elu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(decoder)
    x = Conv2D(filters=INITIAL_FILTER_SIZE // 2, kernel_size=(3, 3), activation="elu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters=INITIAL_FILTER_SIZE, kernel_size=(3, 3), activation="elu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=(3, 3), activation="sigmoid", padding="same")(x)

    model = Model(encoder, decoded, name="autoencoder")
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.summary()
    return model


def get_model(name):
    if name == "convolutional":
        return _build_convolutional()
    elif name == "autoencoder":
        return _build_autoencoder()


def _build_convolutional():
    input_layer = Input(shape=INPUT_SHAPE)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation="elu", padding="same")(input_layer)
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation="elu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    output_layer = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=(3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input_layer, output_layer, name="convolutional")
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def inference(model, image_path):
    image = Image.open(image_path)
    result = model.predict([image])


def train_model(model, data_path, label_path, epochs, batch_size=32):

    @tf.function
    def training_step(x, y):
        # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be
            # recorded on the GradientTape.
            logits = model(x, training=True)

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y, logits)

        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the
        # loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value

    optimizer = model.optimizer
    loss_fn = tf.keras.losses.get(model.loss)

    # load images
    data = []
    labels = []

    print("opening images...")
    for name in tqdm(os.listdir(data_path)[:2000]):
        data.append(np.array(Image.open(os.path.join(data_path, name))))
        labels.append(np.array(Image.open(os.path.join(label_path, name))))

    data = np.asarray(data)
    labels = np.asarray(labels)

    # custom training loop
    print(f"training '{model.name}' for {epochs} epochs...")
    start_time = datetime.now()
    for epoch in range(epochs):
        mini_batch_pos = 0

        # Iterate over the mini-batches of the dataset.
        for _ in trange(len(data), desc=f"Epoch {epoch + 1}"):
            if mini_batch_pos+batch_size < len(data):
                minibatch_data = data[mini_batch_pos:mini_batch_pos + batch_size]
                minibatch_labels = labels[mini_batch_pos:mini_batch_pos + batch_size]
            else:
                minibatch_data = data[-batch_size:]
                minibatch_labels = labels[-batch_size:]
            mini_batch_pos += batch_size

            training_step(np.array(minibatch_data), np.array(minibatch_labels))
    print("total training time:", datetime.now() - start_time)
