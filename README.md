# e-hir GAN Tutorial
GAN implementation using Tensorflow <br />
The Following are software are required to run this code:
1. NVIDIA CUDA latest version
2. Anaconda version 4.3

  After installing Anaconda create environment with python version 3.5
In the environment install the following pakages:

1. keras-gpu
2. pillow 
3. opencv 
4. matplotlib 

The input file should be .mat file including all data. This code is designed to read .mat file with a struct name acitivity orginazed as followes: 
1. name (data type string) 
2. human_number (data type double)
3. data ( matrix 600x140 double) 

Before running the code check the following script and adjust cordingly:
1. GAN_train.py, adjust output directory
2. read.py , adjust input directory

Run GAN_train.py to start the magic.

The code will prodcue 1472 images every 300 epoch. 
