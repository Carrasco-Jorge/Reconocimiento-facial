# Import general modules
import numpy as np
from glob import glob
import os

# Import viz modules
#import matplotlib.pyplot as plt
from PIL import Image

# Used to crop original images. Original images were deleted to save space.
# Please DO NOT try to run this script.

files = glob(f"my_data/*.jpg")
num = 0

cut_h0 = 1000
cut_h1 = 2800
cut_v0 = 500
cut_v1 = -500

for f in files:
    with Image.open(f) as img:
        num += 1
        np_img = np.array(img)
        new_img = Image.fromarray(np_img[cut_h0:cut_h1,cut_v0:cut_v1])
        # Modify images
        new_img.save(f"my_data_png/{num}.png")
print(f"Found {num} images.")
