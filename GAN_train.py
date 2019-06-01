import tensorflow as tf
import numpy as np
import os
import argparse
import glob
import cv2
import scipy
from scipy.misc import imsave
from scipy.misc import imresize
import time


dim = 64
image_size = dim*dim

def read_images(dataset_path):
	Y_data = np.zeros(shape=(1,image_size))
	y = 0
	files = glob.glob (dataset_path)
	y = len(files)
	i = 100000
	for myFile in files:
		image = cv2.imread(myFile,-1)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		i += 1
		image = np.reshape(image, (1,image_size))
		plot(image,1,i)
		if y < 1:
			Y_data = image
		else:
			Y_data = np.vstack((Y_data,image))
	return Y_data, y


def plot(data,type,i):
	if type == 1:
		a = np.zeros((dim,dim))
		sample = data.reshape(dim, dim)
		a[:,:] = sample
		cv2.imwrite('train/%s.png' % str(i).zfill(4), a)

	return

if __name__ == '__main__':

	logdir = '1'
	images_path = 'C:/Users/STUDENT/Desktop/Ibrahim/GAN_tot/images/*.PNG'
	input,n_samples = read_images(images_path)
	
	


