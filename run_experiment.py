import sys
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.append(str(Path(__file__).parent.resolve()))
from model_wrapper import ModelWrapper
from imagenet import read_imagenet, normalize_label


parser = argparse.ArgumentParser()
parser.add_argument('imagenet', type=Path)
parser.add_argument('--model', type=str, default="InceptionV1")
parser.add_argument('--layer', type=str, default="mixed4d")
parser.add_argument('--outer_batch_size', type=int, default=8192)
parser.add_argument('--inner_batch_size', type=int, default=256)
parser.add_argument('--diversity_factor', type=float, default=1e2)
parser.add_argument('--test', action="store_true")
args = parser.parse_args()

date_str = f'{datetime.now():%Y%m%d_%H%M%S}'
Path('output').mkdir(exist_ok=True)
out_dir = Path(f'output/neurons_{date_str}')
out_dir.mkdir(exist_ok=True)
img_dir = out_dir / 'visualizations'
img_dir.mkdir(exist_ok=True)
probs_dir = out_dir / 'probs'
probs_dir.mkdir(exist_ok=True)

out_stats_name = out_dir / f'neurons_{date_str}.csv'
out_stats = open(out_stats_name, 'w')
print('model,layer,feature_ind,importance', file=out_stats, flush=True)

images, true_labels = read_imagenet(args.imagenet)
print(f'{true_labels[:10]=}')
model = ModelWrapper(images, true_labels, args.model,
                     outer_batch_size=args.outer_batch_size, inner_batch_size=args.inner_batch_size)
ls = [l for l in model.model.layers if l.name.startswith(args.layer)]
assert len(ls) == 1
layer = ls[0]
print(f'{layer=}')

print(f"{model.top5_acc=} {model.top10_acc=}")

for ind in range(layer.depth):
    viz = model.visualize(args.layer, ind)
    viz_path = img_dir / f'{normalize_label(args.layer)}_{ind}.png'
    viz.save(viz_path)
    # viz.show()

    if args.test:
        model.test_knockout(args.layer, ind)

    importance, probs_knock = model.importance(args.layer, ind)
    print(f"{layer}:{ind}, {importance=}")

    print(f"{args.model},{layer.name},{ind},{importance}", file=out_stats, flush=True)

    np.save(str(probs_dir / f'{normalize_label(args.layer)}_{ind}_probs_knock.npy'), probs_knock)
