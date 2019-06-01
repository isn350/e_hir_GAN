from read import read_data
from GAN_test import train
import time
import tensorflow as tf
def train_gan(activity):
	logdir = '1'
	data_path = 'C:/Users/STUDENT/Desktop/Ibrahim/GAN_tot/o/' + activity + '/train/' # directly to plot training data for visualization 
	outputdata = 'C:/Users/STUDENT/Desktop/Ibrahim/GAN_tot/o/' # directly to visualization GAN output 
	gan_input,_  = read_data(data_path,[[activity]])
	train(logdir,64,gan_input,outputdata,activity)

if __name__ == '__main__':
	activity = 'boxingmoving'
	train_gan(activity)
