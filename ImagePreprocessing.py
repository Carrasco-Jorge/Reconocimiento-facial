# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:19:29 2022

@author: jorge
"""

# Import general modules
#import numpy as np
from glob import glob
import os

# Import viz modules
#import matplotlib.pyplot as plt
from PIL import Image


for i in range(1,41):
    files = glob(f"data/s{i}/*.pgm")
    num = 0
    for f in files:
        with Image.open(f) as img:
            num += 1
            try:
                img.save(f"data_png/s{i}/{num}.png")
            except:
                os.mkdir(f"./data_png/s{i}")
                img.save(f"data_png/s{i}/{num}.png")
    print(f"Found {num} images.")

