# -*- coding: utf-8 -*-

import time
import png
import numpy as np
from math import sqrt


def save_fig(pixels=[], name=''):
    pixels = np.array(pixels)
    pixels = pixels.reshape(int(sqrt(len(pixels))), int(sqrt(len(pixels))))
    pixels = pixels.tolist()
    s = map(lambda x: map(int, x), pixels)
    name += '/' + str(time.time()) + '.png'
    f = open('generated/' + name, 'wb')
    w = png.Writer(len(s[0]), len(s), greyscale=True, bitdepth=1)
    w.write(f, s)
    f.close()
