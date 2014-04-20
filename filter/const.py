#coding: utf-8

import numpy as np

SOBEL_KERNEL_3_3_X = np.array([
							  	[-1, 0, 1], 
							  	[-2, 0, 2], 
							  	[-1, 0, 1]
							  ])

SOBEL_KERNEL_3_3_Y = np.array([
							  	[1, 2, 1], 
							  	[0, 0, 0], 
							  	[-1, -2, -1]
							  ])


SOBEL_KERNEL_5_5_X = np.array([
								[1, 2, 0, -2, -1], 
								[4, 8, 0, -8, -4], 
								[6, 12, 0, -12, -6], 
								[4, 8, 0, -8, -4], 
								[1, 2, 0, -2, -1]
							  ])

SOBEL_KERNEL_5_5_Y = np.array([
								[-1, -4, -6, -4, -1], 
								[-2, -8, -12, -8, -2], 
								[0, 0, 0, 0, 0], 
								[2, 8, 12, 8, 2], 
								[1, 4, 6, 4, 1]
							  ])

ALLOWED_SOBEL_KERNEL_SIZES = (3, 5)

SOBEL = {
		3: (SOBEL_KERNEL_3_3_X, SOBEL_KERNEL_3_3_Y),
		5: (SOBEL_KERNEL_5_5_X, SOBEL_KERNEL_5_5_Y)
}
