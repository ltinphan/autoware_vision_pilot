from pathlib import Path
import json
from PIL import Image
import numpy as np


def write_detections(detections, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(detections, f, indent=2)


def write_segmentation(seg_map: np.ndarray, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(seg_map.astype('uint8'))
    img.save(out_path)
