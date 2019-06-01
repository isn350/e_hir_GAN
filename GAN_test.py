from read import plot
import tensorflow as tf
import os
import numpy as np

dim=64
image_size = dim*dim
def train(logdir,batch_size,input,dir,activity):
	from model_conv import discriminator, generator


	with tf.variable_scope('placeholder'):
		# Raw image
		X = tf.placeholder(tf.float32, [None, image_size])
		tf.summary.image('raw image', tf.reshape(X, [-1, dim, dim, 1]), 3)
		# Noise
		z = tf.placeholder(tf.float32, [None, 100])	 # noise
		tf.summary.histogram('Noise', z)


	with tf.variable_scope('GAN'):
		G = generator(z,dim, batch_size)

		D_real, D_real_logits = discriminator(X, dim,reuse=False)
		D_fake, D_fake_logits = discriminator(G, dim,reuse=True)
	tf.summary.image('generated image', tf.reshape(G, [-1, dim, dim, 1]), 3)

	with tf.variable_scope('Prediction'):
		tf.summary.histogram('real', D_real)
		tf.summary.histogram('fake', D_fake)

	with tf.variable_scope('D_loss'):
		d_loss_real = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
		d_loss_fake = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
		d_loss = d_loss_real + d_loss_fake

		tf.summary.scalar('d_loss_real', d_loss_real)
		tf.summary.scalar('d_loss_fake', d_loss_fake)
		tf.summary.scalar('d_loss', d_loss)

	with tf.name_scope('G_loss'):
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
								(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))
		tf.summary.scalar('g_loss', g_loss)

	tvar = tf.trainable_variables()
	dvar = [var for var in tvar if 'discriminator' in var.name]
	gvar = [var for var in tvar if 'generator' in var.name]

	with tf.name_scope('train'):
		d_train_step = tf.train.AdamOptimizer(learning_rate=5e-3, beta1=0.5).minimize(d_loss, var_list=dvar)
		g_train_step = tf.train.AdamOptimizer(learning_rate=5e-3, beta1=0.5).minimize(g_loss, var_list=gvar)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter('tmp/'+'gan_conv_'+logdir)
	writer.add_graph(sess.graph)
	
	num_img = 0
	if not os.path.exists('out/'):
		os.makedirs('out/')
	num = 1
	for i in range(3001):
		indices = np.random.choice(144, batch_size)
		X_batch = input[indices]
		batch_X = X_batch
		batch_noise = np.random.uniform(-1., 1., [batch_size, 100])

		if i % 300 == 0:
			dir_output = dir  + '/' +activity + '/'+str(num)+'/'
			for x in range(23):
				sample = sess.run(G, feed_dict={z: np.random.uniform(-1., 1., [64, 100])})
				m=1
				for m in range(64):
					num_img += 1
					x = sample[m,:]	
					plot(x,dir_output,num_img)
					m +=1
			num += 1
		_, d_loss_print = sess.run([d_train_step, d_loss],
								   feed_dict={X: batch_X, z: batch_noise})

		_, g_loss_print = sess.run([g_train_step, g_loss],
								   feed_dict={z: batch_noise})

		if i % 300 == 0:
			s = sess.run(merged_summary, feed_dict={X: batch_X, z: batch_noise})
			writer.add_summary(s, i)
			print('epoch: {}'.format(i))
			print('D loss: {:.4}'. format(d_loss_print))
			print('G_loss: {:.4}'.format(g_loss_print))
			folder = num-1
			print('folder number: {}'.format(folder))
			print()
			data_file = dir + '/' +activity + '/'
			os.chdir(data_file)
			with open("Output.txt", "a") as text_file:
				print('epoch: {}'.format(i),file=text_file)
				print('D loss: {:.4}'.format(d_loss_print),file=text_file)
				print('G_loss: {:.4}'.format(g_loss_print),file=text_file)
				folder = num-1
				print('folder number: {}'.format(folder),file=text_file)
				print('',file=text_file)

				
