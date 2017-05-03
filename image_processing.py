#!/usr/bin/env python


import numpy as np
import cv2
import scipy.signal



def Plop( data, vmax, hmax):
	# vmax, hmax are size of frame
	ans = zeros( (vmax,hmax), float )
	V,H = data.shape
	vctr, hctr = V/2, H/2 # center of frame
	vactr, hactr = vmax/2, hmax/2 # center of blob
	# compute the limits for the answ	
	valo = vactr - vctr
	if valo<0: 
		valo = 0
		vahi = vactr + vctr
	if vahi>=vmax: 
		vahi = vmax
		halo = hactr - hctr
	if halo<0: 
		halo = 0
		hahi = hactr + hctr
	if hahi>=hmax:
		 hahi = hmax
	# compute limits of incoming
		vblo = vctr - vactr
	if vblo<=0: 
		vblo = 0
		vbhi = vctr + vactr
	if vbhi>=V: 
		vbhi= V
		hblo = hctr - hactr
	if hblo<=0: 
		hblo = 0
		hbhi = hctr + hactr
	if hbhi>=H: 
		hbhi = H
	if vahi-valo != vbhi-vblo:
		vbhi = vblo+vahi-valo
	if hahi-halo != hbhi-hblo:
		hbhi = hblo+hahi-halo
	ans[valo:vahi, halo:hahi] = data[vblo:vbhi, hblo:hbhi] + 0
	return ans



def LoadImage(filename):
	img = cv2.imread(filename)
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray_img = gray_img/255.0
	gray_img *= 0.6
	gray_img += 0.4
	output_img =scipy.signal.cspline2d(gray_img,2)
	return output_img




def LoadTarget(filename, data):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = img/255.0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i,j] < 0.9:
				img[i,j]= 1
			else:
				img[i,j]= 0
	targ = Plop(img,data.shape[0],data.shape[1])
	return targ


	
	 
