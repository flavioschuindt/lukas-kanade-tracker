#coding: utf-8

import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageChops

from filter.filter import convert_to_gray_scale, calculate_covariance_matrix, sobel

def create_image_from_pixels(pixels, mode, size):

    im = Image.new(mode, size)
    im.putdata(pixels)
    return im

def communicate_with_ffmpeg_by_pipe(video_file, pix_fmt):

    command = [ 'utils/ffmpeg',
                '-loglevel', 'quiet',
                '-i', video_file,
                '-f', 'image2pipe',
                '-pix_fmt', pix_fmt,
                '-vcodec', 'rawvideo', '-'
    ]

    return subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 8)

def extract_frame_from_video_buffer(buffer, width, height, bytes_per_pixel):

    raw_image = buffer.read(width*height*bytes_per_pixel)
    if raw_image == "":
        return None
    image = np.fromstring(raw_image, dtype='uint8').reshape((height, width)) if bytes_per_pixel == 1 else np.fromstring(raw_image, dtype='uint8').reshape((height, width, 3))
    buffer.flush()
    return image

def diff(f1, f2):
    width, height = f1.size

    f1 = convert_to_gray_scale(f1)
    f2 = convert_to_gray_scale(f2)

    f1 = f1.load()
    f2 = f2.load()

    d = np.zeros((height, width))
    for j in range(height):
        for i in range(width):
            d[j][i] = f1[i,j] - f2[i,j]

    return d

def calc_ix_it_iy_it(j, i, kernel_size, dx, dy, dt):

    middle = kernel_size / 2

    sum_ix_it = 0
    sum_iy_it = 0
    try:
        sum_ix_it = dx[j, i] * dt[j, i]
        sum_iy_it = dy[j, i] * dt[j, i]
    except:
        pass

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

def lukas_kanade_pyramidal(corners, f1_levels, f2_levels, dx, dy, dt, kernel_size=3):
    flow = []
    f1_levels_without_top_and_basis = f1_levels[1:-1]
    f2_levels_without_top_and_basis = f2_levels[1:-1]
    middle = kernel_size / 2
    intermediate_levels_gradient = {}

    # Obtain Ix, Iy, It for intermediate levels
    k = len(f1_levels_without_top_and_basis)
    for f1, f2 in zip(f1_levels_without_top_and_basis, f2_levels_without_top_and_basis):
        dt_level = diff(f1, f2)
        dx_level, dy_level = sobel(f1, kernel_size)
        intermediate_levels_gradient[k] = {'dx':dx_level, 'dy':dy_level, 'dt':dt_level}
        k -= 1


    dt_image_top = diff(f1_levels[0], f2_levels[0])
    dx_image_top, dy_image_top = sobel(f1_levels[0], kernel_size)
    for corner in corners:
        j, i = corner[0], corner[1] # Corner coordinates in the highest resolution level (pyramid basis)
        j_basis, i_basis = j, i

        # Obtain optical flow to lowest resolution (pyramid top)

        # Find correspondent point (i,j) from the highest resolution level in the lowest resolution level 
        j = round(j / (2 ** (len(f1_levels) - 1)))
        i = round(i / (2 ** (len(f1_levels) - 1)))

        c = calculate_covariance_matrix(dx_image_top, dy_image_top, i, j, middle) # covariance matrix for the correspondent point in the lowest resolution

        right = calc_ix_it_iy_it(j, i, kernel_size, dx_image_top, dy_image_top, dt_image_top)
        left = np.array([[c[0][0], c[0][1]], [c[0][1], c[1][1]]])
        try:
            u, v = np.linalg.solve(left, right)
        except:
            flow.append(((i_basis, j_basis), (0,0)))
            continue

        """
            Now, we should propagate this partial flow to the next pyramid level (L - 1).
            To do it, we should follow these steps:
            1 - Obtain correspondent point from the level L into the level L-1. Ex: point (i,j) in level L == (2i,2j) in level L-1.
                Also, remeber that we're transversing the pyramid from top to basis and increasing resolution. So, a point in level L
                maps in 4 points in level L - 1.
            2 - Obtain the optical flow (u',v') around the position (2i+2u, 2j+2v). (u,v) is the optical flow from the pyramid top (partial flow).
            3 - The optical flow in the pyramid level L - 1 is: (2u + u', 2v + v')
            4 - Last, we resize this optical flow (multiply it by a factor of 2) and propagate it to the next level of pyramid.
            5 - Repeat all these steps for all levels until we finish the last level (pyramid basis - image with highest resolution)
        """
        # Obtain optical flow to intermediate resolutions (all resolutions except first and last - top and basis)
        corner_flow = (u,v)
        k = len(f1_levels_without_top_and_basis)
        stop = False
        for f1, f2 in zip(f1_levels_without_top_and_basis, f2_levels_without_top_and_basis):
            u, v = corner_flow
            j = j * 2
            i = i * 2

            dt_level = intermediate_levels_gradient[k]['dt']
            dx_level, dy_level = intermediate_levels_gradient[k]['dx'], intermediate_levels_gradient[k]['dy']
            c = calculate_covariance_matrix(dx_level, dy_level, round(i+2*u), round(j+2*v), middle)

            right = calc_ix_it_iy_it(round(j+2*v), round(i+2*u), kernel_size, dx_level, dy_level, dt_level)
            left = np.array([[c[0][0], c[0][1]], [c[0][1], c[1][1]]])
            try:
                new_u, new_v = np.linalg.solve(left, right)
                corner_flow = (2*u+new_u, 2*v+new_v)
                k -= 1
            except:
                flow.append(((i_basis, j_basis), corner_flow))
                stop = True
                continue

        if stop:
            continue

        # Last resolution (pyramid basis). Final optical flow.
        u, v = corner_flow
        j = j * 2
        i = i * 2
        c = calculate_covariance_matrix(dx, dy, round(i+2*u), round(j+2*v), middle)

        right = calc_ix_it_iy_it(round(j+2*v), round(i+2*u), kernel_size, dx, dy, dt)
        left = np.array([[c[0][0], c[0][1]], [c[0][1], c[1][1]]])
        try:
            new_u, new_v = np.linalg.solve(left, right)
        except:
            flow.append(((i_basis, j_basis), corner_flow))
            continue
        corner_flow = (2*u+new_u, 2*v+new_v)
        flow.append(((i_basis,j_basis), corner_flow))

    return flow
    
def draw_velocity_vector(im, point, vector):
    draw = ImageDraw.Draw(im)
    new_position = (point[0] + vector[0], point[1]+ vector[1])
    draw.line([point, new_position], fill=(0,255,0), width=1)
    draw.point([new_position], fill=(0,0,255))
    del draw
    return im

def right_shift_image(im, offset_x, offset_y):
    return ImageChops.offset(im, offset_x, offset_y)

def get_resolutions(im, levels):
    width, height = im.size
    factor = 2
    images = []
    for i in range(levels):
        im_copy = im.copy()
        im_copy.thumbnail((width/factor, height/factor), Image.ANTIALIAS)
        factor += 2
        images.append(im_copy)

    return images[::-1]


