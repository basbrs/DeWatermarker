import itertools
import os
import cv2
import multiprocessing as mp
import numpy as np
from tqdm import tqdm


def open_data(data_path, label_path, test_split):
    data = []
    labels = []

    names = os.listdir(data_path)[:100]
    train_size = len(names) - int(test_split * len(names))

    print("loading data...")
    for name in tqdm(names):
        data.append(cv2.imread(os.path.join(data_path, name)))
        labels.append(cv2.imread(os.path.join(label_path, name)))

    data = np.array(data)
    labels = np.array(labels)

    shuffle = np.random.permutation(len(names))
    data = data[shuffle] / 255
    labels = labels[shuffle] / 255

    return (data[:train_size], labels[:train_size]), (data[train_size:], labels[train_size:])


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


def resize_images(src, dst, target_dim, rotate=False):
    images = os.listdir(src)
    ratio = target_dim[0] / target_dim[1]

    name_index = 0
    for name in tqdm(images):
        try:
            im = cv2.imread(os.path.join(src, name))
            if im.shape[1] / im.shape[0] == ratio:
                new = cv2.resize(im, dsize=target_dim)
                name_index += 1

                if rotate:
                    new = cv2.rotate(new, cv2.ROTATE_90_CLOCKWISE)

                cv2.imwrite(dst + str(name_index) + ".jpg", new)
        except AttributeError:
            print("AttributeError @", name)
