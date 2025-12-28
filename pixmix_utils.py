# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
IMAGE_SIZE = 32

#########################################################
#################### AUGMENTATIONS ######################
#########################################################


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

#########################################################
######################## MIXINGS ########################
#########################################################

def get_ab(beta):
  if np.random.random() < 0.5:
    a = np.float32(np.random.beta(beta, 1))
    b = np.float32(np.random.beta(1, beta))
  else:
    a = 1 + np.float32(np.random.beta(1, beta))
    b = -np.float32(np.random.beta(1, beta))
  return a, b

def add(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2 - 1, img2 * 2 - 1
  out = a * img1 + b * img2
  return torch.clamp((out + 1) / 2, 0, 1)

def multiply(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2, img2 * 2
  out = (img1 ** a) * (img2.clip(1e-37) ** b)
  return torch.clamp(out / 2, 0, 1)


########################################
##### EXTRA MIXIMGS (EXPREIMENTAL) #####
########################################

def invert(img):
  return 1 - img

def screen(img1, img2, beta):
  img1, img2 = invert(img1), invert(img2)
  out = multiply(img1, img2, beta)
  return invert(out)

def overlay(img1, img2, beta):
  case1 = multiply(img1, img2, beta)
  case2 = screen(img1, img2, beta)
  if np.random.random() < 0.5:
    cond = img1 < 0.5
  else:
    cond = img1 > 0.5
  return torch.where(cond, case1, case2)

def darken_or_lighten(img1, img2, beta):
  if np.random.random() < 0.5:
    cond = img1 < img2
  else:
    cond = img1 > img2
  return torch.where(cond, img1, img2)

def swap_channel(img1, img2, beta):
  channel = np.random.randint(3)
  img1[channel] = img2[channel]
  return img1

#######################3
def high_frequency_mixup(img1, img2, beta):
    a, b = get_ab(beta)
    C, H, W = img1.shape
    range_ratio = 0.6
    center_h, center_w = H // 2, W // 2
    offset_h = int(center_h * range_ratio)
    offset_w = int(center_w * range_ratio)

    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    fft1 = torch.fft.fft2(img1)
    fft2 = torch.fft.fft2(img2)

    for c in range(C):
        high1 = fft1[c, center_h - offset_h:center_h + offset_h, center_w - offset_w:center_w + offset_w]
        high2 = fft2[c, center_h - offset_h:center_h + offset_h, center_w - offset_w:center_w + offset_w]
        fft1[c, center_h - offset_h:center_h + offset_h,
        center_w - offset_w:center_w + offset_w] = a * high1 + b * high2

    out = torch.fft.ifft2(fft1).real
    out = torch.clamp((out + 1) / 2, 0, 1)

    return out


freqmixings = [high_frequency_mixup]
pixmixmixings = [add, multiply]

