#coding: utf-8

from const import SOBEL, MAX_NUMBER_CORNERS, CORNER_RESPONSE_THRESHOLD, HARRIS_STEP_SIZE, HARRIS_SUPRESSION_KERNEL_SIZE
import numpy as np


def convert_to_gray_scale(img):
    return img.convert('L')


def sobel(img, kernel_size):
    if kernel_size in SOBEL:
        gray = convert_to_gray_scale(img)
        pixels = gray.load()
        width, height = gray.size
        dx = np.zeros((height, width))
        dy = np.zeros((height, width))
        middle = kernel_size / 2
        for j in range(height):
            for i in range(width):
                sum_dx = 0
                sum_dy = 0

                sum_dx += pixels[i, j] * SOBEL[kernel_size][0][middle][middle]
                sum_dy += pixels[i, j] * SOBEL[kernel_size][1][middle][middle]

                # right neighboorhood
                for l in range(1, middle + 1):
                    new_j = j+l
                    if 0 <= new_j < height:
                        offset = middle + l
                        sum_dx += pixels[i, new_j] * SOBEL[kernel_size][0][middle][offset]
                        sum_dy += pixels[i, new_j] * SOBEL[kernel_size][1][middle][offset]

                # left neighboorhood
                for l in range(1, middle + 1):
                    new_j = j-l
                    if 0 <= new_j < height:
                        offset = middle - l
                        sum_dx += pixels[i, new_j] * SOBEL[kernel_size][0][middle][offset]
                        sum_dy += pixels[i, new_j] * SOBEL[kernel_size][1][middle][offset]

                # up
                for k in range(middle):
                    p = k + 1
                    new_i = i - p
                    if 0 <= new_i < width:
                        # Immediate up neighboor
                        offset = middle - p
                        sum_dx += pixels[new_i, j] * SOBEL[kernel_size][0][offset][middle]
                        sum_dy += pixels[new_i, j] * SOBEL[kernel_size][1][offset][middle]

                        # up-right
                        for l in range(1, middle + 1):
                            new_j = j + l
                            if 0 <= new_j < height:
                                offset_y = middle + l
                                sum_dx += pixels[new_i, new_j] * SOBEL[kernel_size][0][offset][offset_y]
                                sum_dy += pixels[new_i, new_j] * SOBEL[kernel_size][1][offset][offset_y]

                        # up-left
                        for l in range(1, middle + 1):
                            new_j = j - l
                            if 0 <= new_j < height:
                                offset_y = middle - l
                                sum_dx += pixels[new_i, new_j] * SOBEL[kernel_size][0][offset][offset_y]
                                sum_dy += pixels[new_i, new_j] * SOBEL[kernel_size][1][offset][offset_y]

                # bottom
                for k in range(middle):
                    p = k + 1
                    new_i = i + p
                    if 0 <= new_i < width:
                        # Immediate bottom neighboor
                        offset = middle + p
                        sum_dx += pixels[new_i, j] * SOBEL[kernel_size][0][offset][middle]
                        sum_dy += pixels[new_i, j] * SOBEL[kernel_size][1][offset][middle]

                        # bottom-right
                        for l in range(1, middle + 1):
                            new_j = j + l
                            if 0 <= new_j < height:
                                offset_y = middle + l
                                sum_dx += pixels[new_i, new_j] * SOBEL[kernel_size][0][offset][offset_y]
                                sum_dy += pixels[new_i, new_j] * SOBEL[kernel_size][1][offset][offset_y]

                        # bottom-left
                        for l in range(1, middle + 1):
                            new_j = j - l
                            if 0 <= new_j < height:
                                offset_y = middle - l
                                sum_dx += pixels[new_i, new_j] * SOBEL[kernel_size][0][offset][offset_y]
                                sum_dy += pixels[new_i, new_j] * SOBEL[kernel_size][1][offset][offset_y]

                dx[j][i] = sum_dx
                dy[j][i] = sum_dy

        return (dx, dy)

    return None

def _remove_neighboors(point, points, middle):
    not_removed = [gp for gp in points if abs(gp[0] - point[0]) > middle or abs(gp[1] - point[1]) > middle]
    return (point, not_removed, len(points)-len(not_removed))

def _calculate_covariance_matrix(dx, dy, i, j, middle):

    sum_ix2 = 0
    sum_iy2 = 0
    sum_ix_iy = 0

    # compute harris response

    # ∑Ix², ∑Iy², ∑IxIy 

    sum_ix2 += dx[j, i] ** 2
    sum_iy2 += dy[j, i] ** 2
    sum_ix_iy += dx[j, i] * dy[j, i]

    # right neighboorhood
    for l in range(1, middle + 1):
        try:
            sum_ix2 += dx[j, i + l] ** 2
            sum_iy2 += dy[j, i + l] ** 2
            sum_ix_iy += dx[j, i + l] * dy[j, i + l]
        except:
            pass

    # left neighboorhood
    for l in range(1, middle + 1):
        try:
            sum_ix2 += dx[j, i - l] ** 2
            sum_iy2 += dy[j, i - l] ** 2
            sum_ix_iy += dx[j, i - l] * dy[j, i - l]
        except:
            pass

    # up
    for k in range(middle):
        p = k + 1

        # Immediate up neighboor
        try:
            sum_ix2 += dx[j - p, i] ** 2
            sum_iy2 += dy[j - p, i] ** 2
            sum_ix_iy += dx[j - p, i] * dy[j - p, i]
        except:
            pass

        # up-right
        for l in range(1, middle + 1):
            try:

                sum_ix2 += dx[j - p, i + l] ** 2
                sum_iy2 += dy[j - p, i + l] ** 2
                sum_ix_iy += dx[j - p, i + l] * dy[j - p, i + l]
            except:
                pass

        # up-left
        for l in range(1, middle + 1):
            try:
                sum_ix2 += dx[j - p, i - l] ** 2
                sum_iy2 += dy[j - p, i - l] ** 2
                sum_ix_iy += dx[j - p, i - l] * dy[j - p, i - l]
            except:
                pass

    # bottom
    for k in range(middle):
        p = k + 1

        # Immediate bottom neighboor
        try:
            sum_ix2 += dx[j + p, i] ** 2
            sum_iy2 += dy[j + p, i] ** 2
            sum_ix_iy += dx[j + p, i] * dy[j + p, i]
        except:
            pass

        # bottom-right
        for l in range(1, middle + 1):
            try:
                sum_ix2 += dx[j + p, i + l] ** 2
                sum_iy2 += dy[j + p, i + l] ** 2
                sum_ix_iy += dx[j + p, i + l] * dy[j + p, i + l]
            except:
                pass

        # bottom-left
        for l in range(1, middle + 1):
            try:
                sum_ix2 += dx[j + p, i - l] ** 2
                sum_iy2 += dy[j + p, i - l] ** 2
                sum_ix_iy += dx[j + p, i - l] * dy[j + p, i - l]
            except:
                pass

    return np.array([[sum_ix2, sum_ix_iy], [sum_ix_iy, sum_iy2]])

def _get_corner_response(w):
    return w[0]*w[1] - (0.04*((w[0]+w[1]) ** 2)) #corner response

def harris(dx, dy, width, height, kernel_size):
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
    middle = kernel_size / 2
    good_points = []
    stop = False
    for j in range(height):
        if stop:
            break
        for i in range(0, width, HARRIS_STEP_SIZE):
            c = _calculate_covariance_matrix(dx, dy, i, j, middle) # covariance matrix
            w, v = np.linalg.eig(c) # eigenvalues, eigenvectors
            corner_response = _get_corner_response(w)
            if corner_response > CORNER_RESPONSE_THRESHOLD:
                good_points.append((j, i, min(w), c))
                if len(good_points) == MAX_NUMBER_CORNERS:
                    stop = True
                    break

    # Finally, sort descending and get the maximum value (highest minimun eigenvalue) in a neighboorhood.
    good_points = sorted(good_points, key=lambda x: x[2], reverse=True)

    # Non maximum supression
    to_be_processed = good_points
    point = good_points[0] if len(good_points) > 0 else None
    corners = []
    i = 0
    print "\nEscolhendo pontos Harris...\n"
    middle = HARRIS_SUPRESSION_KERNEL_SIZE / 2
    while point is not None:
        p, to_be_processed, total_removed = _remove_neighboors(point, to_be_processed, middle)
        point = to_be_processed[0] if len(to_be_processed) > 0 else None
        corners.append(p)
        i += total_removed
        print "len_good_points: %d | i: %d | Removidos: %d | Restam: %d" % (len(good_points), i, total_removed, len(to_be_processed))

    print "\nFIM.\n"

    return corners



