import numpy as np
from scipy.misc import imsave
from scipy.misc import imresize
import scipy.io as sio
import scipy.misc
import os
import matplotlib.pyplot as plt


dim = 64
orginal_image_dim_x = 600
orginal_image_dim_y =  140
def plot(data,dir,i):
	data = np.reshape(data, (dim,dim))	
	os.makedirs(os.path.dirname(dir), exist_ok=True)
	os.chdir(dir)
	plt.imsave('%s.png' % str(i).zfill(4), data,cmap="jet")

def images_labels(oct_struct,activityname,dir):
	#read_images_and_gives_images_as_row_and_label_
	number_of_images = oct_struct.size
	for x in range(0, number_of_images):
		y = oct_struct[0,x]	
		answer = y['name']
		z = np.zeros(shape=(1,orginal_image_dim_x*orginal_image_dim_y))
		if answer in activityname:
			z = y['data']
			z = np.reshape(z, (orginal_image_dim_x, orginal_image_dim_y))
			z = scipy.misc.imresize(z, (dim, dim))
			z = np.reshape(z, (1,dim*dim))
			plot(z,dir,x)
			
			if x % 144 == 0:
				features = z
				labels = activityname
			else:
				features = np.vstack((features,z))
				labels = np.vstack((labels,activityname))
	return features, labels

def read_data(dir,activityname):
	data_file = 'C:/Users/STUDENT/Desktop/Ibrahim/GAN_tot' # directory to input data file as mat file 
	os.chdir(data_file)
	mat = sio.loadmat('Saven_activity.mat') # name of the mat file 
	oct_struct = mat['activity']
	#activityname = ['boxingmoving']
	images, labels = images_labels(oct_struct,activityname,dir)
	return images, labels


	
	


