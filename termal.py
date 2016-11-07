# coding: utf-8
#! /usr/bin/python

import os

import numpy as np
from sklearn.ensemble import IsolationForest
from PIL import Image, ImageDraw

#
# core
#

class Patch(object):

    def __init__(self, x, y, v, size):
        self.x = x
        self.y = y
        self.v = v
        self.size = size
        self.p = None # outlier propability

        
def extract_patches(image, patch_size):
    res = []
    width, height = image.size
    for x in xrange(0, width, patch_size):
        for y in xrange(0, height, patch_size):
            image_patch = image.crop((x, y, x + patch_size, y + patch_size))
            res.append(Patch(x, y, list(image_patch.getdata()), patch_size))
    return res


def read_image(path):
    return Image.open(path).convert('L')


def draw_patches(image, patches, f):
    d = ImageDraw.Draw(image, mode=None)
    for e in patches:
        if f(e):
            d.rectangle([e.x, e.y, e.x + e.size, e.y + e.size], fill= "blue")

def main():

    training_paths = ['training/002_001_GRE.tif']
    test_path = 'test/001_001_GRE.tif'
    patch_size = 5
    
    # prep data
    print('getting patches..')
    training_patches = []
    for image in (read_image(path) for path in training_paths):
        training_patches += extract_patches(image, patch_size)

    test_image = read_image(test_path)
    test_patches = extract_patches(test_image, patch_size)

    # outlier detection
    print('outlier detection...')
    training_data = np.array([e.v for e in training_patches])
    clf = IsolationForest(max_samples = 1000)
    clf.fit(training_data)
    for e in test_patches:
        e.p = clf.predict(np.array([e.v]))[0] # should be -1 or 1(=outlier)

    # draw
    print('drawing and saving...')
    def f(x):
        return x.p == -1
    draw_patches(test_image, test_patches, f)
    test_image.save('out.jpg', 'JPEG')

if __name__ == '__main__':
    main()
