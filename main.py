#coding: utf-8
import sys
import glob
import os
from math import sqrt, pow
from time import time

import numpy as np
from PIL import Image

from filter.filter import sobel, harris, convert_to_gray_scale
from utils.utils import create_image_from_pixels, communicate_with_ffmpeg_by_pipe, extract_frame_from_video_buffer, \
						diff, calc_ix_it_iy_it, draw_velocity_vector, right_shift_image, get_resolutions, lukas_kanade_pyramidal
						

if __name__ == "__main__":
	
	path = os.path.join(sys.argv[1], "*.png")
	frames = sorted(glob.glob(path))

	total_frames = len(frames)
	for i in range(total_frames):
		# Open two frames
		f1 = Image.open(frames[i])

		if f1.mode != "RGB":
			f1 = f1.convert("RGB")

		next_frame = i+1
		if next_frame == total_frames:
			break
		f2 = Image.open(frames[next_frame])

		if f2.mode != "RGB":
			f2 = f2.convert("RGB") 

		f1_levels = get_resolutions(f1, 3)
		f2_levels = get_resolutions(f2, 3)
		f1_levels.append(f1)
		f2_levels.append(f2)


		width, height = f1.size

		# Temporal difference between frames
		dt = diff(f1, f2)

		# Apply sobel and harris to get the best points to track

		start = time()
		dx, dy = sobel(f1, 3)
		print "Sobel: %.2f segundos" % (time() - start)

		start = time()
		corners = harris(dx, dy, width, height, 3)
		print "Harris: %.2f segundos" % (time() - start)

		# Obtain optical flow using pyramidal implemenatation of lukas kanade feature tracker
		start = time()
		optical_flow = lukas_kanade_pyramidal(corners, f1_levels, f2_levels, dx, dy, dt, 3)
		for point, velocity_vector in optical_flow:
			f1 = draw_velocity_vector(f1, point, velocity_vector)

		print "Lukas Kanade Pyramidal: %.2f segundos" % (time() - start)
		f1_name = os.path.basename(os.path.splitext(frames[i])[0])
		f2_name = os.path.basename(os.path.splitext(frames[next_frame])[0])

		f1.save(os.path.join(sys.argv[2], "flow_%s_to_%s.png") % (f1_name, f2_name), "png")

'''if __name__ == "__main__":

	pipe = communicate_with_ffmpeg_by_pipe(sys.argv[1], 'rgb24')

	f1 = extract_frame_from_video_buffer(pipe.stdout, 1920, 1080, 3)
	if f1 is not None:
		f1 = Image.fromarray(f1)
	i = 1
	while f1 is not None:	
		f2 = extract_frame_from_video_buffer(pipe.stdout, 1920, 1080, 3)
		if f2 is None:
			break
		f2 = Image.fromarray(f2)

		f1_levels = get_resolutions(f1, 3)
		f2_levels = get_resolutions(f2, 3)
		f1_levels.append(f1)
		f2_levels.append(f2)


		width, height = f1.size

		# Temporal difference between frames
		dt = diff(f1, f2)

		# Apply sobel and harris to get the best points to track

		start = time()
		dx, dy = sobel(f1, 3)
		print "Sobel: %.2f segundos" % (time() - start)

		start = time()
		corners = harris(dx, dy, width, height, 3)
		print "Harris: %.2f segundos" % (time() - start)

		# Obtain optical flow using pyramidal implemenatation of lukas kanade feature tracker
		start = time()
		optical_flow = lukas_kanade_pyramidal(corners, f1_levels, f2_levels, dx, dy, dt, 3)
		for point, velocity_vector in optical_flow:
			f1 = draw_velocity_vector(f1, point, velocity_vector)

		print "Lukas Kanade Pyramidal: %.2f segundos" % (time() - start)
		f1.save("data/walk/flow_%s_to_%s.png" % (i, i+1), "png")

		i += 1
		f1 = f2

	pipe.terminate()'''