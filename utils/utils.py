#coding: utf-8

import subprocess
import numpy
from PIL import Image

def create_image_from_pixels(pixels, mode, size):

    im = Image.new(mode, size)
    im.putdata(pixels)
    return im

def communicate_with_ffmpeg_by_pipe(video_file):

    command = [ 'ffmpeg',
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
    image = numpy.fromstring(raw_image, dtype='uint8').reshape((height, width))
    buffer.flush()
    return image

