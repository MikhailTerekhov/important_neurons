import os
import sys
import cv2
import PIL
from pathlib import Path
import numpy as np

# Removing annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import lucid.optvis.param as param
import lucid.optvis.objectives as objectives
import lucid.modelzoo.vision_models as models
from lucid.modelzoo.get_activations import get_activations_new, get_activations_knockout
import lucid.optvis.render as render

sys.path.append(str(Path(__file__).parent.resolve()))
from imagenet import normalize_label, spaced_concat


def top_k_accuracy(top_l, true_l):
    return np.mean([1 if true_l[i] in top_l[i] else 0 for i in range(top_l.shape[0])])


# Brief interpretations of layers of InceptionV1 can be found at https://distill.pub/2017/feature-visualization/appendix/
class ModelWrapper:
    def __init__(self, images, true_labels):
        self.model = models.InceptionV1()
        self.labels = [normalize_label(l) for l in self.model.labels]
        for i in range(1, 8):
            self.labels.append(f'extra{i}')
        self.images = images
        lo, hi = models.InceptionV1.image_value_range
        self.images_float = images.astype(np.float32) / 255.0 * (hi - lo) + lo
        self.true_labels = true_labels
        self.probs = get_activations_new(self.model, self.images_float, "softmax2")
        print(f'{len(self.labels)=} {self.probs.shape=}')
        inds = np.argsort(self.probs, axis=1)[:, ::-1]

        self.top_predictions = np.array([[self.labels[inds[i, j]]
                                          for j in range(inds.shape[1])]
                                         for i in range(inds.shape[0])])
        self.top5_acc = top_k_accuracy(self.top_predictions[:, :5], self.true_labels)
        self.top10_acc = top_k_accuracy(self.top_predictions[:, :10], self.true_labels)

    def get_activations(self, layer, feature_ind):
        return get_activations_new(self.model, self.images_float, layer, reducer=feature_ind)

    def get_probs_knockout(self, layer, feature_ind, knockout_type="mean"):
        assert knockout_type in ["mean", "zero"]
        if knockout_type == "mean":
            acts = get_activations_new(self.model, self.images_float, layer, reducer=feature_ind)
            avg_act = np.mean(acts)
        elif knockout_type == "zero":
            avg_act = 0.0
        else:
            raise ValueError(f"Unknown knockout type {knockout_type}")
        vals_knock = get_activations_knockout(self.model, self.images_float, layer, feature_ind, avg_act,
                                              "softmax2", outer_batch_size=4096)
        return vals_knock

    def importance(self, layer, feature_ind, top_k=10):
        print(f'Evaluating importance of {layer}:{feature_ind}...', flush=True)
        orig_acc = top_k_accuracy(self.top_predictions[:, :top_k], self.true_labels)
        vals_knock = self.get_probs_knockout(layer, feature_ind)
        top_k_inds = np.argsort(vals_knock, axis=1)[:, -top_k:]
        top_k_labels = np.array([[self.labels[top_k_inds[i, j]] for j in range(top_k)] for i in range(top_k_inds.shape[0])])
        knock_acc = top_k_accuracy(top_k_labels, self.true_labels)
        return orig_acc - knock_acc

    def visualize(self, layer, feature_ind, vis_type="channel", num_generations=4, max_num_images=6):
        if vis_type == "channel":
            obj = objectives.channel(f"{layer}_pre_relu", feature_ind) - 1e2 * objectives.diversity(layer)
        else:
            obj = objectives.neuron(layer, feature_ind) - 1e2 * objectives.diversity(layer)

        p = lambda: param.image(224, batch=num_generations)
        imgs = render.render_vis(self.model, obj, p, verbose=False, use_fixed_seed=True)
        imgs = (imgs[0] * 255).astype(np.uint8)
        imgs = spaced_concat(imgs, 20)

        activations = self.get_activations(f"{layer}_pre_relu", feature_ind)
        activations = activations[:, activations.shape[1] // 2, activations.shape[2] // 2]
        max_inds = np.argsort(activations)[-max_num_images:]
        max_images = self.images[max_inds, ...]
        print(f'{max_images.shape=} {activations.shape=} {max_inds.shape=}')
        max_images = np.array([cv2.resize(img, (224, 224)) for img in max_images])
        max_images = spaced_concat(max_images, 20)

        imgs = np.concatenate([imgs, np.zeros((imgs.shape[0], max_images.shape[1] - imgs.shape[1], imgs.shape[2]),
                                              dtype=np.uint8)], axis=1)
        concat = np.concatenate([imgs, max_images], axis=0)
        return PIL.Image.fromarray(concat)
