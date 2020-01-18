from __future__ import print_function, division
import numpy as np
import cv2
import random
import torch


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        image = np.asarray(image)
        scale = self.output_size / max(image.shape[:2])
        sub_img = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        final_img = np.zeros((self.output_size, self.output_size, 3))
        final_img[:, :, :] = 0
        x_offset = random.randint(0, final_img.shape[0] - sub_img.shape[0])
        y_offset = random.randint(0, final_img.shape[1] - sub_img.shape[1])
        final_img[x_offset:x_offset + sub_img.shape[0], y_offset:y_offset + sub_img.shape[1]] = sub_img
        final_img = np.array(final_img, dtype=np.double)
        final_img = np.divide(final_img, 255)

        return {
            'product_id': sample['product_id'],
            'image': final_img,
            'product_description': sample['product_description']
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {
            'product_id': sample['product_id'],
            'image': torch.tensor(np.array(image, dtype=np.float32)),
            'product_description': sample['product_description'],
        }