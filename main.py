import Model
import Data
import multiprocessing as mp
import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)])

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    mp.freeze_support()
    data_path = "./Watermark/data" #  "C:\\Users\\bbaue\\Downloads\\640x480_resized\\"
    label_path = "./Watermark/labels" #  "C:\\Users\\bbaue\\Downloads\\640x480_watermarked\\"
    images_path = "C:\\Users\\bbaue\\Downloads\\VG_100K\\"

    model = Model.build_convolutional()
    # (x_train, y_train), (x_test, y_test) = Data.open_data(data_path, label_path, .8)

    Model.train_model(model, data_path, label_path, epochs=64, batch_size=32)
    # model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=100, batch_size=4, shuffle=True)
    model.save(".\\models\\" + model.name + ".h5")
