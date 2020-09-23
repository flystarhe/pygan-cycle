import os
import torch

import albumentations as albu
import cv2 as cv
import numpy as np
from pathlib import Path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset

from lxml import etree
from xml.etree import ElementTree


"""
anomaly
├── train
│   ├── anomaly_large
│   │   ├── 1.png
│   │   └── 1.xml
│   ├── anomaly_small
│   │   ├── 2.png
│   │   └── 2.xml
│   ├── contamination
│   │   ├── 3.png
│   │   └── 3.xml
│   └── ok
│       ├── 4.png
│       ├── 4.xml
│       ├── 5.png
│       └── 5.xml
└── val
    ├── anomaly_large
    │   ├── 11.png
    │   └── 11.xml
    ├── anomaly_small
    │   ├── 12.png
    │   └── 12.xml
    ├── contamination
    │   ├── 13.png
    │   └── 13.xml
    └── ok
        ├── 14.png
        └── 14.xml
"""


# pip install -U albumentations
light = albu.Compose([
    albu.OneOf([
        albu.VerticalFlip(p=1.0),
        albu.HorizontalFlip(p=1.0),
    ], p=0.5),
    albu.OneOf([
        albu.JpegCompression(quality_lower=85, quality_upper=100, p=1.0),
        albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.2), p=1.0),
        albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
    ], p=0.5),
], p=1.0)


def do_labelImg(xml_path):
    shapes = []

    if not os.path.isfile(xml_path):
        return shapes

    parser = etree.XMLParser(encoding="utf-8")
    xmltree = ElementTree.parse(xml_path, parser=parser).getroot()
    for object_iter in xmltree.findall("object"):
        bndbox = object_iter.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        shapes.append([xmin, ymin, xmax, ymax])
    return shapes


def check_crop(patch, boxes):
    if boxes.shape[0] == 0:
        return True

    patch = patch.astype(np.float32)
    boxes = boxes.astype(np.float32)

    x_beg = np.maximum(patch[0], boxes[:, 0])
    y_beg = np.maximum(patch[1], boxes[:, 1])
    x_end = np.minimum(patch[2], boxes[:, 2])
    y_end = np.minimum(patch[3], boxes[:, 3])
    overlap_w = np.maximum(x_end - x_beg + 1, 0)
    overlap_h = np.maximum(y_end - y_beg + 1, 0)

    inner = (overlap_w * overlap_h) > 0
    if np.any(inner):
        return False
    return True


def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")


def random_crop(im, size, pad_size=0):
    """Performs random crop (CHW format)."""
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    im_crop = im[:, y: (y + size), x: (x + size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop


def test_crop(im, size):
    """Performs test crop (CHW format)."""
    h, w = im.shape[1:]
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    return [x, y, x + size, y + size]


def crop_image(im, xyxy):
    """Performs crop image (CHW format)."""
    x1, y1, x2, y2 = xyxy
    im_crop = im[:, y1: y2, x1: x2]
    return im_crop


def im2tensor(x):
    # (CHW format) [0, 255] -> [0, 1] -> [-1, 1]
    x = (x / 255.0 - 0.5) * 2
    x = torch.from_numpy(x)
    return x


def tensor2im(x):
    # (CHW format) [-1, 1] -> [0, 1] -> [0, 255]
    x = x.cpu().numpy()
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    x = np.transpose(x, (1, 2, 0))
    x = (x * 0.5 + 0.5) * 255.0
    x = np.clip(x, 0, 255)
    x = x.astype("uint8")
    return x


class AnomalyDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        images = make_dataset(opt.dataroot, opt.max_dataset_size)
        self.crop_size = opt.crop_size
        self.input_nc = opt.input_nc
        self.phase = opt.phase
        self._construct_imdb(images)

    def _construct_imdb(self, images):
        self._imdb = []
        for im_path in sorted(images):
            gt_path = Path(im_path).with_suffix(".xml")
            im_boxes = np.array(do_labelImg(gt_path), dtype=np.float32)
            self._imdb.append({"im_path": im_path, "im_boxes": im_boxes})

    def _rand_another(self, index):
        n = len(self._imdb)
        while True:
            _index = np.random.randint(n)
            if _index != index:
                return _index

    def _prepare_im(self, index, max_iter=10):
        while True:
            im_path = self._imdb[index]["im_path"]
            im = cv.imread(im_path, 1)

            if self.phase == "train":
                im = light(image=im)["image"]
            if self.input_nc == 1:
                im = im[:, :, [0]]
            im = im.astype(np.float32, copy=False)
            # HWC -> CHW
            im = im.transpose((2, 0, 1))
            # Retrieve the im_boxes
            im_boxes = self._imdb[index]["im_boxes"]

            if self.phase == "train":
                assert self.crop_size > 0
                im = zero_pad(im=im, pad_size=32)
                for _ in range(max_iter):
                    xyxy = test_crop(im=im, size=self.crop_size)
                    if check_crop(xyxy, im_boxes):
                        im = crop_image(im=im, xyxy=xyxy)
                        return im, im.copy(), im_path, im_path
            elif self.crop_size > 0:
                im = random_crop(im=im, size=self.crop_size)
                return im, im.copy(), im_path, im_path
            else:
                return im, im.copy(), im_path, im_path

            index = self._rand_another(index)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor)    -- an image in one domain
            A_paths(str) -- the path of the image
        """
        A, B, A_path, B_path = self._prepare_im(index)
        A, B = im2tensor(A), im2tensor(B)  # (CHW format) [-1, 1]
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return len(self._imdb)
