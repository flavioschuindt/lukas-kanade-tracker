#coding: utf-8

from const import SOBEL


def convert_to_gray_scale(img):
    return img.convert('L')


def sobel(img, kernel_size):
    if kernel_size in SOBEL:
        gray = convert_to_gray_scale(img)
        pixels = gray.load()
        width, height = gray.size
        dx = []
        dy = []
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

                dx.append(sum_dx)
                dy.append(sum_dy)

        return (dx, dy)

    return None
