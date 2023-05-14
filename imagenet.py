import os
import sys
import numpy as np
import PIL
import cv2

import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
from imagenet_classes import imagenet_classes


def spaced_concat(images, len_z):
    zeros = np.zeros((images[0].shape[0], len_z, 3), dtype=np.uint8)
    return np.concatenate(sum([[images[i], zeros] for i in range(images.shape[0] - 1)], [])
                          + [images[images.shape[0] - 1, ...]],
                          axis=1)


def top_k_accuracy(top_l, true_l):
    return np.mean([1 if true_l[i] in top_l[i] else 0 for i in range(top_l.shape[0])])


# Normalize a string: remove spaces, convert to lowercase, remove non-alphanumeric characters
def normalize_label(s):
    return ''.join([c for c in s.lower() if c.isalnum()])


# Given a folder, read all images in it into a list of numpy arrays
def read_images_from_dir(dirname):
    images = []
    for filename in tqdm.tqdm(os.listdir(dirname)):
        img = PIL.Image.open(os.path.join(dirname, filename))
        img = np.array(img)
        if img.shape == (64, 64):
            img = np.stack([img, img, img], axis=-1)
        elif img.shape == (64, 64, 3):
            pass
        else:
            print(f'{img.shape=}')
        images.append(img)
    return images


# Read a .npz file containing ImageNet images and labels
def read_imagenet(file_npz):
    images_file = np.load(file_npz)
    true_label_ind = images_file["labels"]
    print(f'{np.min(true_label_ind)=} {np.max(true_label_ind)=}')
    labels = np.array([normalize_label(imagenet_classes[i]) for i in true_label_ind])
    print(f'{labels[:10]=}')
    print(f'{images_file["data"].shape=} {images_file["labels"].shape=}')
    images = np.transpose(images_file["data"].reshape((images_file["data"].shape[0], 3, 64, 64)), (0, 2, 3, 1))
    return images, labels
