2# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import numpy as np
import scipy.ndimage
import scipy.spatial
import skimage.transform
import nibabel as nib

# import medpy.metric

def crop_image(image, cx, cy, size, constant_values=0):
  """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
  X, Y = image.shape[:2]
  rX = int(size[0] / 2)
  rY = int(size[1] / 2)
  x1, x2 = cx - rX, cx + rX
  y1, y2 = cy - rY, cy + rY
  x1_, x2_ = max(x1, 0), min(x2, X)
  y1_, y2_ = max(y1, 0), min(y2, Y)
  # Crop the image
  crop = image[x1_: x2_, y1_: y2_]
  # Pad the image if the specified size is larger than the input image size
  if crop.ndim == 3:
    crop = np.pad(crop,
                  ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                  'constant', constant_values=constant_values)
  elif crop.ndim == 4:
    crop = np.pad(crop,
                  ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                  'constant', constant_values=constant_values)
  else:
    print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
    exit(0)
  return crop


def rescale_intensity(image, thres=(1.0, 99.0)):
  """ Rescale the image intensity to the range of [0, 1] """
  val_l, val_h = np.percentile(image, thres)
  image2 = image
  image2[image < val_l] = val_l
  image2[image > val_h] = val_h
  image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
  return image2


def zero_pad(image):
  (a, b, _) = image.shape
  front = int(np.ceil(np.abs(a - b) / 2.0))
  back = int(np.floor(np.abs(a - b) / 2.0))

  if a > b:
    padding = ((0, 0), (front, back), (0,0))
  else:
    padding = ((front, back), (0, 0), (0,0))

  return np.pad(image, padding, mode='constant', constant_values=0)


def resize_image(image, size, interpolation_order):
  return skimage.transform.resize(image, tuple(size), order=interpolation_order, mode='constant')


def augment_data_2d(whole_image, whole_label, preserve_across_slices, max_shift=10, max_rotate=10, max_scale=0.1):
  new_whole_image = np.zeros_like(whole_image)

  if whole_label is not None:
    new_whole_label = np.zeros_like(whole_label)
  else:
    new_whole_label = None

  for i in range(whole_image.shape[-1]):
    image = whole_image[:, :, i]
    new_image = image

    # For each image slice, generate random affine transformation parameters
    # using the Gaussian distribution
    if preserve_across_slices and i != 0:
      pass
    else:
      shift_val = [np.clip(np.random.normal(), -3, 3) * max_shift,
                   np.clip(np.random.normal(), -3, 3) * max_shift]
      rotate_val = np.clip(np.random.normal(), -3, 3) * max_rotate
      scale_val = 1 + np.clip(np.random.normal(), -3, 3) * max_scale


    new_whole_image[:,:,i] = transform_data_2d(new_image, shift_val, rotate_val, scale_val, interpolation_order=1)

    if whole_label is not None:
      label = whole_label[:, :, i]
      new_label = label
      new_whole_label[:,:,i] = transform_data_2d(new_label, shift_val, rotate_val, scale_val, interpolation_order=0)

  return new_whole_image, new_whole_label


def transform_data_2d(image, shift_value, rotate_value, scale_value, interpolation_order):
  # Apply the affine transformation (rotation + scale + shift) to the image
  row, col = image.shape
  M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_value, 1.0 / scale_value)
  M[:, 2] += shift_value

  return scipy.ndimage.affine_transform(image, M[:, :2], M[:, 2], order=interpolation_order)


def save_nii(image, affine, header, filename):
  if header is not None:
    nii_image = nib.Nifti1Image(image, None, header=header)
  else:
    nii_image = nib.Nifti1Image(image, affine)

  nib.save(nii_image, filename)
  return


def load_nii(nii_image):
  image = nib.load(nii_image)
  affine = image.header.get_best_affine()
  image = image.get_data()

  return image, affine

def data_augmenter(image, label, shift, rotate, scale, intensity, flip):
  """
      Online data augmentation
      Perform affine transformation on image and label,
      which are 4D tensor of shape (N, H, W, C) and 3D tensor of shape (N, H, W).
  """
  image2 = np.zeros(image.shape, dtype=np.float32)
  label2 = np.zeros(label.shape, dtype=np.int32)
  for i in range(image.shape[0]):
    # For each image slice, generate random affine transformation parameters
    # using the Gaussian distribution
    shift_val = [np.clip(np.random.normal(), -3, 3) * shift,
                 np.clip(np.random.normal(), -3, 3) * shift]
    rotate_val = np.clip(np.random.normal(), -3, 3) * rotate
    scale_val = 1 + np.clip(np.random.normal(), -3, 3) * scale
    intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity

    # Apply the affine transformation (rotation + scale + shift) to the image
    row, col = image.shape[1:3]
    M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
    M[:, 2] += shift_val
    for c in range(image.shape[3]):
      image2[i, :, :, c] = ndimage.affine_transform(image[i, :, :, c],
                                                                  M[:, :2], M[:, 2], order=1)

    # Apply the affine transformation (rotation + scale + shift) to the label map
    label2[i, :, :] = ndimage.affine_transform(label[i, :, :],
                                                             M[:, :2], M[:, 2], order=0)

    # Apply intensity variation
    image2[i] *= intensity_val

    # Apply random horizontal or vertical flipping
    if flip:
      if np.random.uniform() >= 0.5:
        image2[i] = image2[i, ::-1, :, :]
        label2[i] = label2[i, ::-1, :]
      else:
        image2[i] = image2[i, :, ::-1, :]
        label2[i] = label2[i, :, ::-1]
  return image2, label2


def np_categorical_dice_2d(pred, truth, num_classes):
  """ Dice overlap metric for label k """

  dice = np.zeros(num_classes)

  for i in range(num_classes):
    pred_class = (pred == i).astype(np.float32)
    truth_class = (truth == i).astype(np.float32)

    # dice[i] = np.nanmean(2 * np.sum(A * B, axis=(0,1)) / (np.sum(A, axis=(0,1)) + np.sum(B, axis=(0,1))))

    per_slice_dice = []
    for j in range(truth.shape[-1]):
      if np.sum(truth_class[:,:,j]) == 0:
        continue
      else:
        per_slice_dice.append(2 * np.sum(truth_class[:,:,j] * pred_class[:,:,j]) / (np.sum(truth_class[:,:,j]) + np.sum(pred_class[:,:,j])))

    # print(per_slice_dice)
    dice[i] = np.mean(per_slice_dice)

  return dice

def np_categorical_dice_3d(pred, truth, num_classes):
  """ Dice overlap metric for label k """

  dice = np.zeros(num_classes)

  for i in range(num_classes):
    A = (pred == i).astype(np.float32)
    B = (truth == i).astype(np.float32)
    dice[i] = 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))

  return dice

def np_categorical_jaccard_3d(pred, truth, num_classes):
  """ Jaccard overlap metric for label k """

  jaccard = np.zeros(num_classes)

  for i in range(num_classes):
    A = (pred == i).astype(np.float32)
    B = (truth == i).astype(np.float32)
    jaccard[i] = np.sum(1.0*np.logical_and(A, B)) / np.sum(1.0*np.logical_or(A,B))

  return jaccard

 
def np_categorical_volume_3d(pred, truth, num_classes, pixel_spacing):
  """ Volume error (percent) for label k """
  volPred = np.zeros(num_classes)
  volTruth = np.zeros(num_classes)
  volE = np.zeros(num_classes)
  volP = np.zeros(num_classes)
 
  for i in range(num_classes):
    volPred[i] = (np.sum(pred == i))*pixel_spacing[0]*pixel_spacing[1]*pixel_spacing[2]*0.001
    volTruth[i] = (np.sum(truth == i))*pixel_spacing[0]*pixel_spacing[1]*pixel_spacing[2]*0.001
    volE[i] = volPred[i] - volTruth[i];
    volP[i] = (volPred[i] - volTruth[i])*1.0 / volTruth[i]*100.0
 
  return volE, volP, volPred, volTruth



'''
def medpy_categorical_dice_2d(pred, truth, num_classes):
  """ Dice overlap metric for label k """

  dice = np.zeros(num_classes)

  for i in range(num_classes):
    dice[i] = medpy.metric.dc(pred == i, truth == i)

  return dice


def medpy_categorical_dice_3d(pred, truth, num_classes):
  """ Dice overlap metric for label k """

  dice = np.zeros(num_classes)

  for i in range(num_classes):
    dice[i] = medpy.metric.dc(pred == i, truth == i)

  return dice
'''

def np_categorical_assd_hd(pred, truth, num_classes, pixel_spacing):
  assd = np.zeros(num_classes)
  hd = np.zeros(num_classes)

  for i in range(num_classes):
    assd[i], hd[i] = distance_metric_2d_average(pred == i, truth == i, pixel_spacing)

  return assd, hd


def np_categorical_assd_hd_3d(pred, truth, num_classes, pixel_spacing):
  assd = np.zeros(num_classes)
  hd = np.zeros(num_classes)

  for i in range(num_classes):
    assd[i], hd[i] = distance_metric_3d(pred == i, truth == i, pixel_spacing)

  return assd, hd

'''
def medpy_categorical_assd_hd_3d(pred, truth, num_classes, pixel_spacing):
  assd = np.zeros(num_classes)
  hd = np.zeros(num_classes)

  for i in range(num_classes):
    assd[i] = medpy.metric.assd(pred == i, truth == i, pixel_spacing)
    hd[i] = medpy.metric.hd(pred == i, truth == i, pixel_spacing)

  return assd, hd


def medpy_categorical_assd_hd(pred, truth, num_classes, pixel_spacing):
  assd = np.zeros(num_classes)
  hd = np.zeros(num_classes)

  for i in range(num_classes):
    assd_list = []
    hd_list = []

    for z in range(pred.shape[-1]):
      slice_A = (pred[:, :, z] == i).astype(np.uint8)
      slice_B = (truth[:, :, z] == i).astype(np.uint8)

      # The distance is defined only when both contours exist on this slice
      if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
        assd_list.append(medpy.metric.assd(slice_A, slice_B, voxelspacing=pixel_spacing))
        hd_list.append(medpy.metric.hd(slice_A, slice_B, voxelspacing=pixel_spacing))
      else:
        assd_list.append(np.nan)
        hd_list.append(np.nan)

    assd[i] = np.nanmean(assd_list)
    hd[i] = np.nanmean(hd_list)

  return assd, hd
'''

def distance_metric_3d(seg_A, seg_B, pixel_spacing):
  X, Y, Z = seg_A.shape

  if np.sum(seg_A.astype(np.uint8)) == 0 or np.sum(seg_B.astype(np.uint8)) == 0:
    return np.nan, np.nan

  pts_A = []
  pts_B = []

  for z in range(Z):
    # Binary mask at this slice
    slice_A = seg_A[:, :, z].astype(np.uint8)
    slice_B = seg_B[:, :, z].astype(np.uint8)

    # Find contours and retrieve all the points
    # contours is a list with length num_contours. Each element is an array with shape (num_points, 1, 2)
    contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
      contours_array = np.concatenate(contours, axis=0)[:,0,:]
      contours_array = np.pad(contours_array, ((0,0),(0,1)), 'constant', constant_values=z)

      pts_A.append(contours_array)

    contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
      contours_array = np.concatenate(contours, axis=0)[:,0,:]
      contours_array = np.pad(contours_array, ((0,0),(0,1)), 'constant', constant_values=z)

      pts_B.append(contours_array)

  pts_A_array = np.concatenate(pts_A, axis=0) * pixel_spacing
  pts_B_array = np.concatenate(pts_B, axis=0) * pixel_spacing

  # Distance matrix between point sets
  # N = np_pairwise_squared_euclidean_distance(pts_A_array, pts_B_array)
  # N = np.sqrt(N)

  # KDTree distance calculation
  d_A2B = getDistancesFromAtoB(pts_A_array, pts_B_array)
  d_B2A = getDistancesFromAtoB(pts_B_array, pts_A_array)

  # Mean distance and hausdorff distance
  # md = 0.5 * (np.mean(np.min(N, axis=0)) + np.mean(np.min(N, axis=1)))
  # hd = np.max([np.max(np.min(N, axis=0)), np.max(np.min(N, axis=1))])

  # KDTree distance calculation
  md = 0.5 * (np.mean(d_A2B, axis=0) + np.mean(d_B2A, axis=0))
  # hd = np.max([np.max(d_A2B, axis=0), np.max(d_B2A, axis=0)])
  hd = np.max([np.max(np.percentile(d_A2B, 95), axis=0), np.max(np.percentile(d_B2A, 95), axis=0)])

  return md, hd

def distance_metric_2d_average(seg_A, seg_B, pixel_spacing):
  """
      Measure the distance errors between the contours of two segmentations.
      The manual contours are drawn on 2D slices.
      We calculate contour to contour distance for each slice.
      """
  table_md = []
  table_hd = []
  # seg_A = np.expand_dims(seg_A, axis=-1)
  # seg_B = np.expand_dims(seg_B, axis=-1)
  X, Y, Z = seg_A.shape
  for z in range(Z):
    # Binary mask at this slice
    slice_A = seg_A[:, :, z].astype(np.uint8)
    slice_B = seg_B[:, :, z].astype(np.uint8)

    # The distance is defined only when both contours exist on this slice
    if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
      # slice_A = get_largest_connected_component(slice_A).astype(np.uint8)
      # slice_B = get_largest_connected_component(slice_B).astype(np.uint8)
      # Find contours and retrieve all the points
      contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                        cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_NONE)

      pts_A = np.concatenate(contours, axis=0)[:,0,:] * pixel_spacing

      contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                        cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_NONE)

      pts_B = np.concatenate(contours, axis=0)[:,0,:] * pixel_spacing

      # Distance matrix between point sets
      N = np_pairwise_squared_euclidean_distance(pts_A, pts_B)
      N = np.sqrt(N)

      # Distance matrix between point sets
      # M = np.zeros((len(pts_A), len(pts_B)))
      # for i in range(len(pts_A)):
      #   for j in range(len(pts_B)):
      #     M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

      # print(np.allclose(M, N, rtol=1e-5, atol=1e-5))

      # Mean distance and hausdorff distance
      md = 0.5 * (np.mean(np.min(N, axis=0)) + np.mean(np.min(N, axis=1)))
      hd = np.max([np.max(np.min(N, axis=0)), np.max(np.min(N, axis=1))])
      # md = np.mean(np.min(N, axis=1))
      # hd = np.max(np.min(N, axis=1))
      table_md += [md]
      table_hd += [hd]
    else:
      table_md += [np.nan]
      table_hd += [np.nan]

  # Return the mean distance and Hausdorff distance across 2D slices
  mean_md = np.nanmean(table_md) if table_md else None
  mean_hd = np.nanmean(table_hd) if table_hd else None

  return mean_md, mean_hd


def np_pairwise_squared_euclidean_distance(x, z):
  '''
  This function calculates the pairwise euclidean distance
  matrix between input matrix x and input matrix z and
  return the distances matrix as result.

  x is a BxN matrix
  z is a CxN matrix
  result d is a BxC matrix that contains the Euclidean distances

  '''
  # Calculate the square of both
  x_square = np.expand_dims(np.sum(np.square(x), axis=1), axis=1)
  z_square = np.expand_dims(np.sum(np.square(z), axis=1), axis=0)

  # Calculate x*z
  x_z = np.matmul(x, np.transpose(z))

  # Calculate squared Euclidean distance
  d_matrix = x_square + z_square - 2 * x_z
  d_matrix[d_matrix < 0] = 0

  return d_matrix

def get_largest_connected_component(binary_mask):
   labels = skimage.measure.label(binary_mask, background=0)
   bin_count = np.bincount(labels.flat).astype(np.float32)
   bin_count[0] = np.nan
   return labels == np.nanargmax(bin_count)


def getDistancesFromAtoB(a, b):
  kdTree = scipy.spatial.KDTree(a, leafsize=100)
  return kdTree.query(b, k=1, eps=0, p=2)[0]
