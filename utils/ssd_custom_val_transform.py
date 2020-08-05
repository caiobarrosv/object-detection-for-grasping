"""Transforms described in https://arxiv.org/abs/1512.02325."""
import numpy as np
import mxnet as mx
import gluoncv.data.transforms.bbox as tbbox
import gluoncv.data.transforms.image as timage

'''
Code taken and modified from the official gluoncv repository
'''

class SSDCustomValTransform(object):
    """Default SSD training transform which includes tons of image augmentations."""

    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        if anchors is None:
            return

        # since we do not have predictions yet, so we ignore sampling here
        from gluoncv.model_zoo.ssd.target import SSDTargetGenerator
        self._target_generator = SSDTargetGenerator(
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=-1, **kwargs)

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize with random interpolation
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        bbox = tbbox.resize(label, (w, h), (self._width, self._height))

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        cls_targets, box_targets, _ = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0]