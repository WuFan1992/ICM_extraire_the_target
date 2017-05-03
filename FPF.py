#!/usr/bin/env python


import numpy as np
import cv2
from matplotlib import pyplot as plt


def FPF(img,c,fp):
	#get the length and the width of img, 
	height = img.shape[0]
	width =  img.shape[1]
	# img must be a grid_scale image and normalising
	D = (pow(abs(img),fp)).sum(1) # sum(0)means that we calcul the sum for each coloum and finaly we forme a 1 x width array
	D = D/width
	   

	for i in range(height):
		if D[i]< 0.001:
			D[i] = 0.001*(np.sign(D[i]+1e9))  
	#now D is a 1x width matrix
	#Y = (img/np.sqrt(D))
	D_all = np.zeros((height,height))
	for j in range(height):
		for i in range(height):
			D_all[j][i] = D[j]
	D_inv = np.linalg.inv(D_all)
	print (D_inv)
	Y = np.dot(np.sqrt(D_inv),img)
	Yc = Y.transpose()
	Q1 = np.dot(Yc,Y)
	# if height = 1 , it means that Q is just a value not a matrix	
	#if img.shape[1]:
	Q = np.linalg.inv(Q1)
	#else:
	
	#Q = 1/Q
	Rc = np.dot(Q,c)
	H = np.dot(np.sqrt(D_inv),np.dot(Y,Rc))
	return H



if __name__ == '__main__':
	img = cv2.imread("house.jpg")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img/255.0
	#X = np.zeros((1,img.shape[0]*img.shape[1]))
	#X = (np.fft.fft2(img)).ravel()
	X = np.fft.fft2(img)
	#height = len(img)
	cst = np.ones((img.shape[1],img.shape[1]))
	#filt = X
	filt = FPF(X,cst,0.3)
	print (X)
	print (filt)
	#filt = X.reshape((img.shape[0],img.shape[1]))
	#d_shift = np.array(np.dstack([filt.real,filt.imag]))
	filt = np.fft.ifft2(filt)*img.shape[0]*img.shape[1]
	real = filt.real
	imag = filt.imag
	img_final = np.sqrt(real*real + imag*imag)*255.0
	print(img_final)
	#img_back = cv2.idft(d_shift)*img.shape[0]*img.shape[1]
	#img_final = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
	#img_final = cv2.magnitude(filt[:,:,0],filt[:,:,1])
	
	plt.imshow(img_final, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()
	#cv2.imshow('image',img)
	#cv2.waitKey(0)                 
	#cv2.destroyAllWindows()	
	
	
	
	

		
		
	
			
	
