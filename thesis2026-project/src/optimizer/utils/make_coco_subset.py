import random
import shutil
from pathlib import Path

random.seed(42)

PATH = "../datasets/"
src = Path(PATH + "/coco/train/images/")
dst = Path(PATH + "coco_subset/train_1percent/images")
print(f"src: {src}")
print(f"dst: {dst}")
dst.mkdir(parents=True, exist_ok=True)

imgs = list(src.glob("*.jpg"))
subset = random.sample(imgs, int(len(imgs) * 0.01))  # 1%
print(len(subset))

for img in subset:
    shutil.copy(img, dst / img.name)