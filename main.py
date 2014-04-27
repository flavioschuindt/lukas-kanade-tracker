#coding: utf-8
import sys
from math import sqrt, pow

import numpy as np
from PIL import Image

from filter.filter import sobel, harris, convert_to_gray_scale
from utils.utils import create_image_from_pixels, communicate_with_ffmpeg_by_pipe, extract_frame_from_video_buffer, \
						diff, calc_ix_it_iy_it, draw_velocity_vector
						

if __name__ == "__main__":

	# Open two frames
	f1 = Image.open(sys.argv[1])
	f2 = Image.open(sys.argv[2])

	print f1.mode
	print f2.mode

	# Obtain difference between frames
	dt = diff(f1, f2)

	# Apply sobel and harris to get the best points to track
	width, height = f1.size
	print "Inicio Sobel"
	dx, dy = sobel(f1, 3)
	print "Fim Sobel"
	corners = harris(dx, dy, width, height, 3)

	for corner in corners:
		j, i, min_w, c = corner
		right = calc_ix_it_iy_it(j, i, 3, dx, dy, dt)
		left = np.array([[c[0][0], c[0][1]], [c[0][1], c[1][1]]])
		u, v = np.linalg.solve(left, right)
		print "O ponto (%d, %d) tem o vetor velocidade (%f, %f) e portanto sua localização no frame 2 é (%f, %f)" % (j, i, u, v, round(j+v), round(i+u))
		f1 = draw_velocity_vector(f1, (j, i), (v, u))

	f1.save("data/flow.png", "png")

'''if __name__ == "__main__":


	pipe = communicate_with_ffmpeg_by_pipe(sys.argv[1], 'rgb24')

	i = 0
	while True:
		frame = extract_frame_from_video_buffer(pipe.stdout, 1920, 800, 3)
		if frame is None:
			break
		i += 1
		result = Image.fromarray(frame)
		result.save("data/frames_RGB/frame_%s.png" % str(i), "png")

	pipe.terminate()'''

'''if __name__ == "__main__":

	f1 = Image.open(sys.argv[1])
	f2 = Image.open(sys.argv[2])

	d = diff(f1, f2)

	result = Image.fromarray(d)
	result = result.convert('RGB')
	result.save("data/diff.png", "png")'''

'''if __name__ == "__main__":

	im = Image.open(sys.argv[1])

	dx, dy = sobel(im, 3)

	width, height = im.size

	image_with_sobel_applied = []

	for j in range(height):
		for i in range(width):
			mag = sqrt(pow(dx[j][i], 2) + pow(dy[j][i], 2))
			image_with_sobel_applied.append(mag) if mag <= 255 else image_with_sobel_applied.append(255)

	result = create_image_from_pixels(image_with_sobel_applied, 'L', im.size)
	result.save("data/sobel.png", "png")'''

'''if __name__ == "__main__":

	"""from timeit import Timer
	im = Image.open(sys.argv[1])

	t = Timer(lambda: harris(im, 3))
	print t.timeit(number=1)"""

	im = Image.open(sys.argv[1])
	width, height = im.size
	print "Inicio Sobel"
	dx, dy = sobel(im, 3)
	print "Fim Sobel"
	corners = harris(dx, dy, width, height, 3)
	pixels = im.load()

	for corner in corners:
		j, i, min_w, c = corner
		pixels[i, j] = (0, 255, 0)

	im.save("data/harris_from_blur.png", "png")'''