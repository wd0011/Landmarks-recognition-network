# -*- coding: utf-8 -*-
'''
This file is going to resize image and reload it to dataset

'''
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageOps
import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub
import json

origin_list = glob.glob('data/*.jpg')
new_width = 256
new_height = 256

for i in range(len(origin_list)):
    # print(data_list[i])
    # response = open(data_list[i],'r+')
    pil_image = Image.open(origin_list[i])
    # print(pil_image)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb.save('test/%d.jpg'%(i), format='JPEG', quality=90)


