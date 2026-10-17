import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def decode_mask(record, root):
    if "mask" in record:
        mask = np.asarray(Image.open(root / record["mask"]).convert("L")) != 0
    else:
        segmentation = record["segmentation"]
        if isinstance(segmentation, dict):
            rles = segmentation
        elif isinstance(segmentation[0], dict):
            rles = segmentation
        else:
            rles = coco_mask.frPyObjects(segmentation, record["height"], record["width"])
        mask = coco_mask.decode(rles)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
    return np.asarray(mask, dtype=np.uint8)


def bbox_from_mask(mask):
    y, x = np.nonzero(mask)
    if len(y) == 0:
        return '{"bbox_2d": [0,0,0,0]}'
    h, w = mask.shape
    box = [round(x.min() / w * 1000), round(y.min() / h * 1000),
           round(x.max() / w * 1000), round(y.max() / h * 1000)]
    return '{"bbox_2d": [' + ','.join(str(int(v)) for v in box) + ']}'


def foreground_patch_indices(mask, grid, merge_size=2):
    _, h, w = [int(x) for x in grid]
    y, x = np.nonzero(mask)
    ty = ((y / mask.shape[0] * h) // merge_size).astype(np.int64)
    tx = ((x / mask.shape[1] * w) // merge_size).astype(np.int64)
    return np.unique(ty * (w // merge_size) + tx)
