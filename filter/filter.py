#coding: utf-8

from const import SOBEL
import numpy as np


def convert_to_gray_scale(img):
    return img.convert('L')


def sobel(img, kernel_size):
    if kernel_size in SOBEL:
        gray = convert_to_gray_scale(img)
        pixels = gray.load()
        width, height = gray.size
        dx = np.zeros((width, height))
        dy = np.zeros((width, height))
        middle = kernel_size / 2
        for j in range(height):
            for i in range(width):
                sum_dx = 0
                sum_dy = 0

                sum_dx += pixels[i, j] * SOBEL[kernel_size][0][middle][middle]
                sum_dy += pixels[i, j] * SOBEL[kernel_size][1][middle][middle]

                # right neighboorhood
                for l in range(1, middle + 1):
                    try:
                        sum_dx += pixels[i, j + l] * SOBEL[kernel_size][0][middle][middle + l]
                        sum_dy += pixels[i, j + l] * SOBEL[kernel_size][1][middle][middle + l]
                    except:
                        pass

                # left neighboorhood
                for l in range(1, middle + 1):
                    try:
                        sum_dx += pixels[i, j - l] * SOBEL[kernel_size][0][middle][middle - l]
                        sum_dy += pixels[i, j - l] * SOBEL[kernel_size][1][middle][middle - l]
                    except:
                        pass

                # up
                for k in range(middle):
                    p = k + 1

                    # Immediate up neighboor
                    try:
                        sum_dx += pixels[i - p, j] * SOBEL[kernel_size][0][middle - p][middle]
                        sum_dy += pixels[i - p, j] * SOBEL[kernel_size][1][middle - p][middle]
                    except:
                        pass

                    # up-right
                    for l in range(1, middle + 1):
                        try:
                            sum_dx += pixels[i - p, j + l] * SOBEL[kernel_size][0][middle - p][middle + l]
                            sum_dy += pixels[i - p, j + l] * SOBEL[kernel_size][1][middle - p][middle + l]
                        except:
                            pass

                    # up-left
                    for l in range(1, middle + 1):
                        try:
                            sum_dx += pixels[i - p, j - l] * SOBEL[kernel_size][0][middle - p][middle - l]
                            sum_dy += pixels[i - p, j - l] * SOBEL[kernel_size][1][middle - p][middle - l]
                        except:
                            pass

                # bottom
                for k in range(middle):
                    p = k + 1

                    # Immediate bottom neighboor
                    try:
                        sum_dx += pixels[i + p, j] * SOBEL[kernel_size][0][middle + p][middle]
                        sum_dy += pixels[i + p, j] * SOBEL[kernel_size][1][middle + p][middle]
                    except:
                        pass

                    # bottom-right
                    for l in range(1, middle + 1):
                        try:
                            sum_dx += pixels[i + p, j + l] * SOBEL[kernel_size][0][middle + p][middle + l]
                            sum_dy += pixels[i + p, j + l] * SOBEL[kernel_size][1][middle + p][middle + l]
                        except:
                            pass

                    # bottom-left
                    for l in range(1, middle + 1):
                        try:
                            sum_dx += pixels[i + p, j - l] * SOBEL[kernel_size][0][middle + p][middle - l]
                            sum_dy += pixels[i + p, j - l] * SOBEL[kernel_size][1][middle + p][middle - l]
                        except:
                            pass

                dx[i][j] = sum_dx
                dy[i][j] = sum_dy

        return (dx, dy)

    return None

def _remove_neighboors(point, points, middle):
    not_removed = [gp for gp in points if abs(gp[0] - point[0]) > middle or abs(gp[1] - point[1]) > middle]
    return (point, not_removed, len(points)-len(not_removed))


def harris(img, kernel_size):
    """
        Given a kernel_size of size Q, calculate the correlation matrix C for each pixel:

        C = [ 
              ∑Ix²       ∑IxIy
              ∑IxIy      ∑Iy²
            ]

        After, calculate corner response R based on eigenvalues of correlation matrix C:

        R = Det(C) - k(Trace(C))² With k = 0.06
        if R > 1000, it's a good good point.

    """
    width, height = img.size
    dx, dy = sobel(img, kernel_size)
    middle = kernel_size / 2
    good_points = []
    for j in range(height):
        for i in range(width):
            sum_ix2 = 0
            sum_iy2 = 0
            sum_ix_iy = 0

            # compute harris response

            # ∑Ix², ∑Iy², ∑IxIy 

            sum_ix2 += dx[i, j] ** 2
            sum_iy2 += dy[i, j] ** 2
            sum_ix_iy += dx[i, j] * dy[i, j]

            # right neighboorhood
            for l in range(1, middle + 1):
                try:
                    sum_ix2 += dx[i, j + l] ** 2
                    sum_iy2 += dy[i, j + l] ** 2
                    sum_ix_iy += dx[i, j + l] * dy[i, j + l]
                except:
                    pass

            # left neighboorhood
            for l in range(1, middle + 1):
                try:
                    sum_ix2 += dx[i, j - l] ** 2
                    sum_iy2 += dy[i, j - l] ** 2
                    sum_ix_iy += dx[i, j - l] * dy[i, j - l]
                except:
                    pass

            # up
            for k in range(middle):
                p = k + 1

                # Immediate up neighboor
                try:
                    sum_ix2 += dx[i - p, j] ** 2
                    sum_iy2 += dy[i - p, j] ** 2
                    sum_ix_iy += dx[i - p, j] * dy[i - p, j]
                except:
                    pass

                # up-right
                for l in range(1, middle + 1):
                    try:

                        sum_ix2 += dx[i - p, j + l] ** 2
                        sum_iy2 += dy[i - p, j + l] ** 2
                        sum_ix_iy += dx[i - p, j + l] * dy[i - p, j + l]
                    except:
                        pass

                # up-left
                for l in range(1, middle + 1):
                    try:
                        sum_ix2 += dx[i - p, j - l] ** 2
                        sum_iy2 += dy[i - p, j - l] ** 2
                        sum_ix_iy += dx[i - p, j - l] * dy[i - p, j - l]
                    except:
                        pass

            # bottom
            for k in range(middle):
                p = k + 1

                # Immediate bottom neighboor
                try:
                    sum_ix2 += dx[i + p, j] ** 2
                    sum_iy2 += dy[i + p, j] ** 2
                    sum_ix_iy += dx[i + p, j] * dy[i + p, j]
                except:
                    pass

                # bottom-right
                for l in range(1, middle + 1):
                    try:
                        sum_ix2 += dx[i + p, j + l] ** 2
                        sum_iy2 += dy[i + p, j + l] ** 2
                        sum_ix_iy += dx[i + p, j + l] * dy[i + p, j + l]
                    except:
                        pass

                # bottom-left
                for l in range(1, middle + 1):
                    try:
                        sum_ix2 += dx[i + p, j - l] ** 2
                        sum_iy2 += dy[i + p, j - l] ** 2
                        sum_ix_iy += dx[i + p, j - l] * dy[i + p, j - l]
                    except:
                        pass



            c = np.array([[sum_ix2, sum_ix_iy], [sum_ix_iy, sum_iy2]]) # correlation matrix
            w, v = np.linalg.eig(c) # eigenvalues, eigenvectors
            corner_response = w[0]*w[1] - (0.06*((w[0]+w[1]) ** 2)) # corner response
            if corner_response > 1000000000000:
                good_points.append((i, j, min(w)))

    # Finally, sort descending and get the maximum value (highest minimun eigenvalue) in a neighboorhood.
    good_points = sorted(good_points, key=lambda x: x[2], reverse=True)
    '''i = 0
    len_good_points = len(good_points)
    while i < len_good_points:
        current_point = good_points[i]
        good_points = good_points[0:i] + [gp for gp in good_points if abs(gp[0] - current_point[0]) > middle or abs(gp[1] - current_point[1]) > middle]
        len_good_points = len(good_points)
        i += 1'''

    to_be_processed = good_points
    point = good_points[0] if len(good_points) > 0 else None
    corners = []
    i = 0
    while i < len(good_points):
        p, to_be_processed, total_removed = _remove_neighboors(point, to_be_processed, middle)
        point = to_be_processed[0] if len(to_be_processed) > 0 else None
        corners.append(p)
        i += total_removed
        print "len_good_points: %d | i: %d | Removidos: %d | Restam: %d" % (len(good_points), i, total_removed, len(to_be_processed))

    return corners



