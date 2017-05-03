#!/usr/bin/env python


import numpy as np
import cv2
import icm
import target_processing
import image_processing
import PCECorrelate
import Peaks


def SingleIteration(icm,data,filt,targ):
	
	icm.IterateLS(data)
	edge = image_processing.EnhanceEdge(icm.Y)
	corr = abs(PCECorrelate.PCECorrelation(edge,filt))
	pce = PCECorrelate.PEAK_CORRELATION_ENERGY(corr)
	pks = Peaks.Peaks(pce,targ)
	data, mask = Peaks.Enhance_peak(data,pks,targ)
	all_one = np.ones(data.shape)
	icm.T = mask*0.9*icm.T + (all_one-mask)*icm.T
	return data


def Driver(filename,targetname):
	data = image_processing.LoadImage(filename)
	target = target_processing.LoadTarget(targetname)
	filt = target_processing.EdgeEncourage(targ)
	filt = target_processing.NormFilter(filt)
	icm = icm.ICM(data.shape)
	for i in range (15):
		print ("iteration: %d" %i)
		data = SingleIteration(icm,data,filt,targ)
	return data
	
