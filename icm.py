#!/usr/bin/env python

import numpy as np
import scipy.signal
import cv2
import sympy.geometry as ge



def LevelSet(A):

	# in the paper F1,A is the OFF function 
	# in the paper F2,A is the ON function 
	# M(A) is the smooth function ,so we use csline2d as the smooth function
	#Aofftarg = A <= 0    # if the pixel is 0 in A ,it will become 1 in Aofftarg, so the image inverse
	Aofftarg = np.zeros(A.shape)
	for y in range(A.shape[0]):
		for x in range(A.shape[1]):
			if A[y,x]==0:
				Aofftarg[y,x] = 1

	Aontarg = A.copy()       # image maintain
	M = np.zeros(A.shape)
	Mofftarg = np.zeros(A.shape)
	M = scipy.signal.cspline2d(Aontarg,70)
	# if we want to check whether the original OFF(0) is from A 
	Mofftarg = M*Aofftarg
	#Aadd = (Mofftarg>0.5)
	Aadd = np.zeros(A.shape)
	for y in range(A.shape[0]):
		for x in range(A.shape[1]):
			if Mofftarg[y,x]>0.5:
				Aadd[y,x] = 1
	A = A + Aadd
	
	#Aontarg = A >0
	Aontarg = A.copy() 
	Montarg = np.zeros(A.shape)
	M = scipy.signal.cspline2d(Aontarg,70)
	Montarge = M*Aontarg+ Aofftarg
	#Akill = (Montarge<0.5)
	Akill = np.zeros(A.shape)
	for y in range(A.shape[0]):
		for x in range(A.shape[1]):
			if Montarge[y,x]<0.5:
				Akill[y,x] = 1
	A = A - Akill
	return A


	
def Circle( size, loc,rad):
	b1,b2 = np.indices( size )
	b1,b2 = b1-loc[0], b2-loc[1]
	mask = b1*b1 + b2*b2
	mask = mask <= rad*rad
	return mask.astype(int)	
	




class ICM:
	# intersecting cortical model
	f,t1,t2 = 0.9,0.8,20.0
	
	def __init__(self,dim):
		self.F = np.zeros(dim,float)
		self.Y = np.zeros(dim,float)
		self.T = np.ones(dim,float)

	def Iterate (self,stim):
		if self.Y.sum() > 0:
			work = scipy.signal.cspline2d(self.Y.astype(float),3)
		else:
			work = np.zeros(self.Y.shape,float)
		self.F = self.f * self.F + stim + 8*work
		for y in range(self.F.shape[0]):
			for x in range(self.F.shape[1]):
				if self.F[y,x]>self.T[y,x]:
					self.Y[y,x]=1
				else:
					self.Y[y,x]=0
		self.T = self.t1 * self.T + self.t2 * self.Y + 0.1



	def IterateLS(self,stim):
		if sum(sum(self.Y))>10:
			work=LevelSet(self.Y)
			work=LevelSet(work)
		else:
			work= np.zeros(self.Y.shape,float)
	
		self.F = self.f*self.F + stim + work
		for y in range(self.F.shape[0]):
			for x in range(self.F.shape[1]):
				if self.F[y,x]>self.T[y,x]:
					self.Y[y,x]=1
				else:
					self.Y[y,x]=0
		self.T = self.t1*self.T+self.t2*self.Y + 0.1



def Corner(data):
	'''
	this function receives the pulse image and returns an edge and corner enhanced image
	'''

	if data.sum()>10:
		a = scipy.signal.cspline2d(data,2)
		corners = np.exp(-(a-data))
		#corners = a
	else:
		corners = np.zeros(data.shape)
	return corners


def PeakDetect(img):
	'''
	this function return the list of coordonne of peak pixel

	'''
	max_value = img.max()
	#print(max_value)
	height = img.shape[0]
	width = img.shape[1]

	peaks= []
	ok = 0
	# max_value must be bigger than 0.9 because we want the image not very dark
	if max_value>0.9 :
		ok = 1
	# while is the circle to detect peaks points
	while ok:
		v,h = divmod(img.argmax(),width)  # argmax returns the indice of one line so if we want to get the (x,y),we must divide H
		if img[v,h]<0.5*max_value:
			ok = 0
			
		else:
			peaks.append((v,h))
			circ = Circle(img.shape,(v,h),10)
			img*=np.ones(img.shape)-circ
	return img,peaks






def Mark(marks,peaks):
	'''
	this function put the peaks points 
	
	'''
	for v,h in peaks:
		marks[v-2:h+3,h] = 1
		marks[v,h-2:h+3] = 1


def Mix(marks,gray_img):
	'''

	this function mix the peaks points with 

	'''
	background = gray_img/gray_img.max()
	background*=250
	
	final_img = (np.ones(gray_img.shape)-marks)*background + marks*255
	return final_img
	



if __name__ == '__main__':
	
	
	img = cv2.imread('input.png')
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#ret,binary_img = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
	#binary_img = binary_img/255
	gray_img = gray_image/255.0
	print(gray_img.max())		
	
	icm = ICM(gray_img.shape)

	marks = np.zeros(gray_img.shape)
	icm.IterateLS(gray_img)
	icm.IterateLS(gray_img)
	icm.IterateLS(gray_img)
	icm.Y = np.ones(icm.Y.shape) - icm.Y
	corners = Corner(icm.Y)
	img_peak,peaks = PeakDetect(corners)
	Mark(marks,peaks)
	final_img = Mix(marks,gray_img)

	'''
	img = np.zeros((360,360))
	img[120:160,120:240]=1
	img[160:240,160:200]=1
	icm = ICM(img.shape)
	'''
	#print (corners-img).max())
	cv2.imshow('Y',img_peak)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	


