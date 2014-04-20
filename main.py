#coding: utf-8
import sys
from math import sqrt, pow

from PIL import Image

from filter.filter import sobel
from utils.utils import create_image_from_pixels


if __name__ == "__main__":

    im = Image.open(sys.argv[1])

    dx, dy = sobel(im, 3)

    width, height = im.size

    image_with_sobel_applied = []

    for i in range(width*height):
    	mag = sqrt(pow(dx[i], 2) + pow(dy[i], 2))
    	image_with_sobel_applied.append(mag) if mag <= 255 else image_with_sobel_applied.append(255)

    result = create_image_from_pixels(image_with_sobel_applied, 'L', im.size)
    result.save("img/sobel.png", "png")