import Model
import Data
import multiprocessing as mp


if __name__ == "__main__":
    mp.freeze_support()
    resized_path = "C:\\Users\\bbaue\\Downloads\\640x480_resized\\"
    watermarked_path = "C:\\Users\\bbaue\\Downloads\\640x480_watermarked\\"

    model = Model.build_convolutional()
    (x_train, y_train), (x_test, y_test) = Data.open_data(watermarked_path, resized_path, .8)

    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64, shuffle=True)
    model.save(".\\models\\" + model.name + ".h5")
