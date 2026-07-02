"""Online 2D affine data augmentation used during training."""
import numpy as np
import cv2
from scipy import ndimage
from numpy.linalg import inv

__all__ = ['augment_data_batch']


def augment_data_batch(image, label, shift=30, rotate=60, scale=0.2, intensity=0.2, flip=True):
    aug_image = np.zeros_like(image)
    aug_label = np.zeros_like(label)
    for i in range(image.shape[0]):
        tem_image = image[i, :, :]
        tem_label = label[i, :, :]
        
        if flip:
            if np.random.rand() > 0.5:
                tem_image = np.fliplr(tem_image)
                tem_label = np.fliplr(tem_label)
            if np.random.rand() > 0.5:
                tem_image = np.flipud(tem_image)
                tem_label = np.flipud(tem_label)

        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        shift_val = [int(shift * np.random.uniform(-1, 1)),
                     int(shift * np.random.uniform(-1, 1))]
        rotate_val = rotate * np.random.uniform(-1, 1)
        scale_val = 1 + scale * np.random.uniform(-1, 1)

        tem_image = tem_image * (1 + intensity * np.random.uniform(-1, 1))

        # Apply the affine transformation (rotation + scale + shift) to the image
        row, col = tem_image.shape
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
        M[:, 2] += shift_val
        M = np.concatenate((M, np.array([[0, 0, 1]])), axis=0)

        aug_image[i, :, :] = ndimage.affine_transform(tem_image, inv(M), order=1)
        aug_label[i, :, :] = ndimage.affine_transform(tem_label, inv(M), order=0)

    return aug_image, aug_label
