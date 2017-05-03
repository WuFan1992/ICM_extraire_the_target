#!/usr/bin/env python


import numpy as np
import cv2



def SWAP(frequence_img):
	'''
	After we do the FFT, the high frequence is in the middle which contains the details of
	a image. we just want the edge so we must remove the high frequence in the middle 
	
	'''
	
	height = frequence_img.shape[0]
	width = frequence_img.shape[1]

	new_img = np.zeros((height,width))
	
	new_img[0:height/2,0:width/2] = frequence_img[height/2:height,width/2:width]
	new_img[0:height/2,width/2:width] = frequence_img[height/2:height,0:width/2]
	new_img[height/2:height,width/2:width] = frequence_img[0:height/2,0:width/2]
	new_img[height/2:height,0:width/2] = frequence_img[0:height/2,width/2:width]
	return new_img


def Correlate(img1,img2):
	'''
	img1 and img2 are the two pictures which we want to compare the similarity

	'''
	img1_fre = np.fft.fft2(img1)
	img2_fre = np.fft.fft2(img2)
	Corr_fre = img1_fre*img2_fre.conjugate()
	Corr = np.fft.ifft2(Corr_fre)
	Corr = SWAP(Corr)
	return Corr

def PEAK_CORRELATION_ENERGY(img,rad):
	
	energy_img = abs(img)**2
	
	kernel = np.ones((rad,rad))
	relative_puisssance = signal.convolve2d(energy_img,kernel,boundary = 'symm',mode = 'same')
	
	output = img / relative_puissance
	return output


def PCECorrelation(edge,filt):
	Corr = Correlate(edge,filt)
	if Corr.max()>0.001:
		pce = PEAK_CORRELATION_ENERGY(Corr,2400) # why 2400?
		return pce
	else:
		return Corr



		
	
