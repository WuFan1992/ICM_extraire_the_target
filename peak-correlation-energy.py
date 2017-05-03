#!/usr/bin/env python

import numpy as np
import cv2
import signal

def PEAK_CORRELATION_ENERGY(img,rad):
	
	energy_img = abs(img)**2
	
	kernel = np.ones((rad,rad))
	relative_puisssance = signal.convolve2d(energy_img,kernel,boundary = 'symm',mode = 'same')
	
	output = energy_img / relative_puissance
	return output
