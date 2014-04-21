#coding: utf-8
import sys
from math import sqrt, pow

from PIL import Image

from filter.filter import sobel
from utils.utils import create_image_from_pixels
from utils.utils import communicate_with_ffmpeg_by_pipe, extract_frame_from_video_buffer


if __name__ == "__main__":

	try:

		pipe = communicate_with_ffmpeg_by_pipe(sys.argv[1])

		i = 0
		while True:
			frame = extract_frame_from_video_buffer(pipe.stdout, 1920, 800)
			if frame is None:
				break
			i += 1
			result = Image.fromarray(frame)
			result.save("data/frames/frame_%s.png" % str(i), "png")

	finally:

		pipe.terminate()

	'''im = Image.open(sys.argv[1])

	dx, dy = sobel(im, 3)

	width, height = im.size

	image_with_sobel_applied = []

	for i in range(width*height):
		mag = sqrt(pow(dx[i], 2) + pow(dy[i], 2))
		image_with_sobel_applied.append(mag) if mag <= 255 else image_with_sobel_applied.append(255)

	result = create_image_from_pixels(image_with_sobel_applied, 'L', im.size)
	result.save("img/sobel.png", "png")'''