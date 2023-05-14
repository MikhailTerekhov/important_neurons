import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.resolve()))
from model_wrapper import ModelWrapper
from imagenet import read_imagenet


parser = argparse.ArgumentParser()
parser.add_argument('imagenet', type=Path)
parser.add_argument('--layer', type=str, default="mixed4e")
args = parser.parse_args()

images, true_labels = read_imagenet(args.imagenet)
model = ModelWrapper(images, true_labels)
print(f"{model.top5_acc=} {model.top10_acc=}")

for ind in range(10):
    print(f'{args.layer}:{ind}')
    viz = model.visualize(args.layer, ind)
    viz.show()

    importance = model.importance(args.layer, ind)
    print(f"{importance=}")


