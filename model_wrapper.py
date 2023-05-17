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
from imagenet import normalize_label, normalize_label_set, spaced_concat


def top_k_accuracy(top_l, true_l, normalize=True):
    vals = [1 if true_l[i] in top_l[i] else 0 for i in range(top_l.shape[0])]
    return np.mean(vals) if normalize else np.sum(vals)


def top_k_accuracy_set(top_l, true_l, normalize=True):
    vals = [1 if true_l[i] in set.union(*top_l[i, :]) else 0 for i in range(top_l.shape[0])]
    return np.mean(vals) if normalize else np.sum(vals)


# Brief interpretations of layers of InceptionV1 can be found at https://distill.pub/2017/feature-visualization/appendix/
# I don't know of other resources that contain comprehensive feature visualizations now that Microscope is not available
class ModelWrapper:
    def __init__(self, images, true_labels, model="InceptionV1",
                 outer_batch_size=8192, inner_batch_size=256, diversity_factor=1e2):
        self.model_name = model
        self.outer_batch_size = outer_batch_size
        self.inner_batch_size = inner_batch_size
        self.diversity_factor = diversity_factor
        self.true_labels = true_labels
        self.images_float = images.astype(np.float32) / 255.0
        if self.model_name == "InceptionV1":
            self.output_layer = "softmax2"
            self.model = models.InceptionV1()
            self.labels = [{normalize_label(l)} for l in self.model.labels]
            # This instance of InceptionV1 has 8 extra output classes, reason unknown
            for i in range(1, 8):
                self.labels.append({f'extra{i}'})
            lo, hi = models.InceptionV1.image_value_range
        elif self.model_name == "InceptionV4_slim":
            self.output_layer = "InceptionV4/Logits/Predictions"
            self.model = models.InceptionV4_slim()
            self.labels = [normalize_label_set(l, true_labels) for l in self.model.labels]
            # lo, hi = models.InceptionV4_slim.image_value_range
            # [0, 1] seems to work significantly better than the recommended [-1, 1]
            lo, hi = 0, 1
        elif self.model_name == "VGG19_caffe":
            self.output_layer = "prob"
            self.model = models.VGG19_caffe()
            self.labels = [normalize_label_set(l, true_labels) for l in self.model.labels]
            self.images_float = self.images_float[:, :, :, ::-1]
            lo, hi = 0, 1
        else:
            raise ValueError(f"Unknown model {self.model_name}")
        self.images_float = self.images_float * (hi - lo) + lo

        # print(f'{set(self.labels) - set(true_labels)=}')
        # print(f'{set(true_labels) -set(self.labels)=}')
        self.images = images
        # self.images_float = images.astype(np.float32) / 255.0
        self.probs = get_activations_new(self.model, self.images_float, self.output_layer,
                                         batch_size=self.inner_batch_size,
                                         outer_batch_size=self.outer_batch_size)
        print(f'{len(self.labels)=} {self.probs.shape=}')
        inds = np.argsort(self.probs, axis=1)[:, ::-1]

        # A 2D array of sets; each set corresponds to the labels of a single image on the given logit
        self.top_predictions = np.array([[self.labels[inds[i, j]]
                                          for j in range(inds.shape[1])]
                                         for i in range(inds.shape[0])], dtype=object)
        print(f'{self.top_predictions[:10, :5]=}')
        self.top5_acc = top_k_accuracy_set(self.top_predictions[:, :5], self.true_labels)
        self.top10_acc = top_k_accuracy_set(self.top_predictions[:, :10], self.true_labels)

    def get_activations(self, layer, feature_ind):
        return get_activations_new(self.model, self.images_float, layer, feature_ind=feature_ind)

    def get_probs_knockout(self, layer, feature_ind, knockout_type="mean"):
        assert knockout_type in ["mean", "zero", "check"]
        if knockout_type == "mean":
            acts = get_activations_new(self.model, self.images_float, layer, feature_ind=feature_ind,
                                       batch_size=self.inner_batch_size,
                                       outer_batch_size=self.outer_batch_size)
            avg_act = np.mean(acts)
        elif knockout_type == "zero":
            avg_act = 0.0
        elif knockout_type == "check":
            acts = get_activations_new(self.model, self.images_float, layer, feature_ind=feature_ind,
                                       batch_size=self.inner_batch_size,
                                       outer_batch_size=self.outer_batch_size)
            avg_act = acts
        else:
            raise ValueError(f"Unknown knockout type {knockout_type}")
        vals_knock = get_activations_knockout(self.model, self.images_float, layer, feature_ind, avg_act,
                                              self.output_layer,
                                              batch_size=self.inner_batch_size,
                                              outer_batch_size=self.outer_batch_size)
        return vals_knock

    def test_knockout(self, layer, feature_ind):
        probs_knock = self.get_probs_knockout(layer, feature_ind, "check")
        assert np.allclose(probs_knock, self.probs)

    def importance(self, layer, feature_ind, top_k=10):
        print(f'Evaluating importance of {layer}:{feature_ind}...', flush=True)
        orig_acc = top_k_accuracy_set(self.top_predictions[:, :top_k], self.true_labels, normalize=False)
        probs_knock = self.get_probs_knockout(layer, feature_ind)
        top_k_inds = np.argsort(probs_knock, axis=1)[:, -top_k:]
        top_k_labels = np.array([[self.labels[top_k_inds[i, j]] for j in range(top_k)] for i in range(top_k_inds.shape[0])])
        knock_acc = top_k_accuracy_set(top_k_labels, self.true_labels, normalize=False)

        return orig_acc - knock_acc, probs_knock

    def visualize(self, layer, feature_ind, vis_type="channel", num_generations=4, max_num_images=8):
        cur_layer = f"{layer}_pre_relu" if self.model_name == "InceptionV1" else layer
        if vis_type == "channel":
            obj = objectives.channel(cur_layer, feature_ind) \
                  - self.diversity_factor * objectives.diversity(layer)
        else:
            obj = objectives.neuron(layer, feature_ind) \
                  - self.diversity_factor * objectives.diversity(layer)

        p = lambda: param.image(self.model.image_shape[0], self.model.image_shape[1], batch=num_generations)
        imgs = render.render_vis(self.model, obj, p, verbose=False, use_fixed_seed=True)
        imgs = (imgs[0] * 255).astype(np.uint8)
        imgs = spaced_concat(imgs, 20)

        activations = self.get_activations(cur_layer, feature_ind)
        activations = activations[:, activations.shape[1] // 2, activations.shape[2] // 2]
        max_inds = np.argsort(activations)[-max_num_images:]
        max_images = self.images[max_inds, ...]
        print(f'{max_images.shape=} {activations.shape=} {max_inds.shape=}')
        max_images = np.array([cv2.resize(img, self.model.image_shape[1::-1]) for img in max_images])
        max_images = spaced_concat(max_images, 20)

        imgs = np.concatenate([imgs, np.zeros((imgs.shape[0], max_images.shape[1] - imgs.shape[1], imgs.shape[2]),
                                              dtype=np.uint8)], axis=1)
        concat = np.concatenate([imgs, max_images], axis=0)
        return PIL.Image.fromarray(concat)
