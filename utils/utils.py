#coding: utf-8

from PIL import Image

def create_image_from_pixels(pixels, mode, size):

    im = Image.new(mode, size)
    im.putdata(pixels)
    return im