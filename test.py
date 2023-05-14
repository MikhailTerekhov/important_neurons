import sys
import os

# Removing annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import PIL
import cv2

import tqdm
from pathlib import Path
import argparse

import lucid.optvis.param as param
import lucid.optvis.objectives as objectives
import lucid.modelzoo.vision_models as models
from lucid.modelzoo.get_activations import get_activations_new, get_activations_knockout
import lucid.optvis.render as render

sys.path.append(str(Path(__file__).parent.resolve()))
from imagenet_classes import imagenet_classes

tf.compat.v1.disable_eager_execution()

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# Given a folder, read all images in it into a list of numpy arrays
def read_images(dirname):
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


def spaced_concat(images, len_z):
    zeros = np.zeros((images[0].shape[0], len_z, 3), dtype=np.uint8)
    return np.concatenate(sum([[images[i], zeros] for i in range(images.shape[0] - 1)], [])
                          + [images[images.shape[0] - 1, ...]],
                          axis=1)


def resize_x4(images):
    B, H, W, C = images.shape
    vals = np.zeros((B, 4 * H, 4 * W, C), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            vals[:, i::4, j::4, :] = images
    return vals


def ablated_classes(model, layer, feature_ind):
    act = get_activations_new(model, layer)


def top_k_accuracy(top_l, true_l):
    return np.mean([1 if true_l[i] in top_l[i] else 0 for i in range(top_l.shape[0])])


# Normalize a string: remove spaces, convert to lowercase, remove non-alphanumeric characters
def normalize_label(s):
    return ''.join([c for c in s.lower() if c.isalnum()])


parser = argparse.ArgumentParser()
parser.add_argument('imagenet_dir', type=Path)
args = parser.parse_args()

model = models.InceptionV1()
model_labels = [normalize_label(l) for l in model.labels]
print(f'{model_labels[:10]=}')
# print(model.labels)

if args.imagenet_dir.suffix == '.npz':
    imagenet = np.load(args.imagenet_dir)
    true_label_ind = imagenet["labels"]
    print(f'{np.min(true_label_ind)=} {np.max(true_label_ind)=}')
    true_labels = np.array([normalize_label(imagenet_classes[i]) for i in true_label_ind])
    print(f'{true_labels[:10]=}')
    print(f'{imagenet["data"].shape=} {imagenet["labels"].shape=}')
    imagenet = np.transpose(imagenet["data"].reshape((imagenet["data"].shape[0], 3, 64, 64)), (0, 2, 3, 1))

    # For each label in model_labels, get the index of this label in label_names_data
else:
    imagenet = read_images(str(args.imagenet_dir))
    # stack images in imagenet along the zeroth axis, convert to floats, to tensorflow tensors, and normalize
    imagenet = np.stack(imagenet, axis=0)

# Print the difference between sets of labels in both directions
# print(f'{set(model_labels) - set(true_labels)=}')
# print(f'{set(true_labels) -set(model_labels)=}')
assert set(model_labels) - {'dummy'} == set(true_labels)

# resized = []
# for i in tqdm.tqdm(range(imagenet.shape[0])):
#     resized.append(cv2.resize(imagenet[i, ...], (224, 224)))
# imagenet_resized = np.concatenate(resized, axis=0)
# del resized
# imagenet_resized = imagenet_resized.astype(np.float32)

# TODO: what's up with image scaling? Seems like it doesn't matter
lo, hi = models.InceptionV1.image_value_range
print(f'{lo=} {hi=}')
imagenet_preproc = imagenet.astype(np.float32)
imagenet_preproc = imagenet_preproc / 255.0 * (hi - lo) + lo
# imagenet_preproc = imagenet_preproc / 255.0
# imagenet_preproc = imagenet_preproc / 255.0 * 2 - 1
print(f'{imagenet_preproc.shape=}')

# PIL.Image.fromarray(np.concatenate(imagenet[0:10, ...], axis=1)).show()

layer_name = "mixed4e"
num_generations = 4
num_samples = 6

acts = get_activations_new(model, imagenet_preproc, "mixed4e", reducer=0)
avg_act = np.mean(acts)
vals_knock = get_activations_knockout(model, imagenet_preproc, "mixed4e", 0, avg_act,
                                      "softmax2", outer_batch_size=4096)
print(f"{vals_knock.shape=}")

top_prediction_inds = np.argsort(vals_knock, axis=1)[:, -10:]
top_predictions = np.array([[model_labels[top_prediction_inds[i, j]]
                             for j in range(top_prediction_inds.shape[1])]
                            for i in range(top_prediction_inds.shape[0])])
print(f"{top_predictions[:10,:]=}")
acc_knock = top_k_accuracy(top_predictions, true_labels)
print(f'{acc_knock=}')

vals = get_activations_new(model, imagenet_preproc, "softmax2", outer_batch_size=4096)
print(f'{np.sum(vals)=}')
print(f"{vals.shape=}")

top_prediction_inds = np.argsort(vals, axis=1)[:, -10:]
top_predictions = np.array([[model_labels[top_prediction_inds[i, j]]
                             for j in range(top_prediction_inds.shape[1])]
                            for i in range(top_prediction_inds.shape[0])])
print(f"{top_predictions[:10,:]=}")

acc = top_k_accuracy(top_predictions, true_labels)

print(f'{acc=} {acc_knock=}')

# TODO rescale imagenet according to model?
print(model.layers)
for _ in range(10):
    ind = np.random.randint(0, 500)
    print(ind)

    activations = get_activations_new(model, imagenet_preproc, layer_name, mean_axes=(1, 2))
    print(f'{activations.shape=}')
    activations = activations[:, ind]
    # Find the top indices of the activations
    max_inds = np.argsort(activations)[-num_samples:]
    # Get the images corresponding to those indices
    max_images = imagenet[max_inds, ...]
    max_images = np.array([cv2.resize(img, (224, 224)) for img in max_images])
    max_images = spaced_concat(max_images, 20)

    obj = objectives.channel(f"{layer_name}_pre_relu", ind) - 1e2 * objectives.diversity(layer_name)
    objn = objectives.neuron(f"{layer_name}_pre_relu", ind) - 1e2 * objectives.diversity(layer_name)

    p = lambda: param.image(224, batch=num_generations)
    imgs = render.render_vis(model, obj, p, verbose=False, use_fixed_seed=True)
    imgs = (imgs[0] * 255).astype(np.uint8)
    imgs = spaced_concat(imgs, 20)
    imgs = np.concatenate([imgs, np.zeros((imgs.shape[0], max_images.shape[1] - imgs.shape[1], imgs.shape[2]),
                                          dtype=np.uint8)], axis=1)
    concat = np.concatenate([imgs, max_images], axis=0)

    print(f'{concat.shape=}')
    PIL.Image.fromarray(concat).show()
