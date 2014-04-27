#coding: utf-8

import subprocess
import numpy as np
from PIL import Image

def create_image_from_pixels(pixels, mode, size):

    im = Image.new(mode, size)
    im.putdata(pixels)
    return im

def communicate_with_ffmpeg_by_pipe(video_file):

    command = [ 'utils/ffmpeg',
                '-loglevel', 'quiet',
                '-i', video_file,
                '-f', 'image2pipe',
                '-pix_fmt', 'gray',
                '-vcodec', 'rawvideo', '-'
    ]

    return subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 8)

def extract_frame_from_video_buffer(buffer, width, height):

    raw_image = buffer.read(width*height)
    if raw_image == "":
        return None
    image = np.fromstring(raw_image, dtype='uint8').reshape((height, width))
    buffer.flush()
    return image

def diff(f1, f2):
    width, height = f1.size
    f1 = f1.load()
    f2 = f2.load()

    d = np.zeros((height, width))
    for j in range(height):
        for i in range(width):
            d[j][i] = f1[i,j] - f2[i,j]

    return d

def calc_ix_it_iy_it(j, i, kernel_size, dx, dy, dt):

    middle = kernel_size / 2

    sum_ix_it = dx[j, i] * dt[j, i]
    sum_iy_it = dy[j, i] * dt[j, i]

    # right neighboorhood
    for l in range(1, middle + 1):
        try:
            sum_ix_it += dx[j, i + l] * dt[j, i + l]
            sum_iy_it += dy[j, i + l] * dt[j, i + l]
        except:
            pass

    # left neighboorhood
    for l in range(1, middle + 1):
        try:
            sum_ix_it += dx[j, i - l] * dt[j, i - l]
            sum_iy_it += dy[j, i - l] * dt[j, i - l]
        except:
            pass

    # up
    for k in range(middle):
        p = k + 1

        # Immediate up neighboor
        try:
            sum_ix_it += dx[j - p, i] * dt[j - p, i]
            sum_iy_it += dy[j - p, i] * dt[j - p, i]
        except:
            pass

        # up-right
        for l in range(1, middle + 1):
            try:
                sum_ix_it += dx[j - p, i + l] * dt[j - p, i + l]
                sum_iy_it += dy[j - p, i + l] * dt[j - p, i + l]
            except:
                pass

        # up-left
        for l in range(1, middle + 1):
            try:
                sum_ix_it += dx[j - p, i - l] * dt[j - p, i - l]
                sum_iy_it += dy[j - p, i - l] * dt[j - p, i - l]
            except:
                pass

    # bottom
    for k in range(middle):
        p = k + 1

        # Immediate bottom neighboor
        try:
            sum_ix_it += dx[j + p, i] * dt[j + p, i]
            sum_iy_it += dy[j + p, i] * dt[j + p, i]
        except:
            pass

        # bottom-right
        for l in range(1, middle + 1):
            try:
                sum_ix_it += dx[j + p, i + l] * dt[j + p, i + l]
                sum_iy_it += dy[j + p, i + l] * dt[j + p, i + l]
            except:
                pass

        # bottom-left
        for l in range(1, middle + 1):
            try:
                sum_ix_it += dx[j + p, i - l] * dt[j + p, i - l]
                sum_iy_it += dy[j + p, i - l] * dt[j + p, i - l]
            except:
                pass

    return np.array([-1 * sum_ix_it, -1 * sum_iy_it])
    
