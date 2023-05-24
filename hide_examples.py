import argparse
import cv2
import numpy as np
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('data', type=Path)
args = parser.parse_args()

# iterate over all images in the args.data, and load them using cv2.imread
for img_path in (args.data / "visualizations").iterdir():
    img = cv2.imread(str(img_path))
    img[img.shape[0] // 2:, :, :] = 0
    cv2.imwrite(str(img_path.with_stem(img_path.stem + "_hidden")), img)
