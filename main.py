import Model
import itertools
import cv2
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm


def _watermark_single(args):
    name, (src, dst, watermark) = args
    image = cv2.imread(os.path.join(src, name))
    height, width = image.shape[:-1]
    pos = (np.random.randint(0, watermark.shape[1] - width), np.random.randint(0, watermark.shape[0] - height))
    watermark_cut = watermark[pos[1]:pos[1] + height, pos[0]:pos[0] + width]
    beta = np.random.random() / 2 + .5
    cv2.imwrite(os.path.join(dst, name), cv2.addWeighted(image, 1, watermark_cut, beta, 0))


def watermark_images(src, dst, watermark):
    names = os.listdir(src)
    params = zip(names, itertools.repeat((src, dst, watermark), len(names)))

    with mp.Pool(mp.cpu_count()) as p:
        list(tqdm(p.imap(_watermark_single, params), total=len(names)))


def resize_images(src, dst, target_dim):
    images = os.listdir(src)
    ratio = target_dim[0] / target_dim[1]

    name_index = 0
    for name in tqdm(images):
        try:
            im = cv2.imread(os.path.join(src, name))
            if im.shape[1] / im.shape[0] == ratio:
                new = cv2.resize(im, dsize=target_dim)
                name_index += 1
                cv2.imwrite(dst + str(name_index) + ".jpg", new)
        except AttributeError:
            print("AttributeError @", name)


if __name__ == "__main__":
    mp.freeze_support()
    width = 640
    height = 480
    resized_path = "C:\\Users\\bbaue\\Downloads\\640x480_resized\\"
    watermarked_path = "C:\\Users\\bbaue\\Downloads\\640x480_watermarked\\"
    watermark_hq = cv2.imread("watermark_small.png")

    # watermark_images("C:\\Users\\bbaue\\Downloads\\640x480_resized\\",
    #                  "C:\\Users\\bbaue\\Downloads\\640x480_watermarked\\",
    #                  watermark_hq)
    # resize_images("C:\\Users\\bbaue\\Downloads\\VG_100K_2", (640, 480),
    #               "C:\\Users\\bbaue\\Downloads\\640x480_resized\\")

    model = Model.build_convolutional()
    (x_train, y_train), (x_test, y_test) = Model.open_data(watermarked_path, resized_path, .8)

