# Import general modules
import numpy as np
from glob import glob
import os

# Import viz modules
import matplotlib.pyplot as plt
from PIL import Image

# Used to crop original images. Original images were deleted to save space.
# Please DO NOT try to run this script.

files = glob(f"test_data/me/*.jpg")
num = 0

cut_h0 = 700
cut_h1 = -700
cut_v0 = 1300
cut_v1 = -800

func = lambda img: Image.fromarray(img[cut_v0:cut_v1,cut_h0:cut_h1])

for f in files:
    with Image.open(f) as img:
        num += 1
        np_img = np.array(img)
        new_img = func(np_img)
        # Modify images
        new_img.save(f"test_data_png/me/{num}.png")
print(f"Found {num} images in 'me'.")

# ------------------------------------------------------------------- #

files = glob(f"test_data/not_me/*.jpg")
num = 0

for f in files:
    with Image.open(f) as img:
        num += 1
        np_img = np.array(img)
        new_img = func(np_img)
        # Modify images
        new_img.save(f"test_data_png/not_me/{num}.png")
print(f"Found {num} images in 'not_me'.")
