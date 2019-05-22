import numpy as np;
import math;
import random;

from mnist import MNIST
mndata = MNIST('/home/sthithpragya/ARTIN_lab/lab3/python-mnist/data')
images, labels = mndata.load_training() 
#images contains the pixel data for each of the 60k images
#labels contains the correct value of the number in the image

images_norm = np.array(images)/255.0; #normalised image data 
images_norm = np.transpose(images_norm);
labels = np.array(labels);
n_examples = len(labels); #total number of examples (60k)

# loading the test data
images_test, labels_test = mndata.load_testing() 

images_test_norm = np.array(images_test)/255.0;
images_test_norm = np.transpose(images_test_norm);
labels_test = np.array(labels_test);
n_test_examples = len(labels_test); # = 10k

alpha = 0.05;

#initialising weight and bias matrices/ vectors
n_0 = 784;
n_1 = 30;
n_2 = 10;
size_1 = (30,784);
size_2 = (10,30);

w_1 = np.random.normal(0,1/np.sqrt(n_0),size_1); #(centre,standard deviation,size)
w_2 = np.random.normal(0,1/np.sqrt(n_1),size_2); #(centre,standard deviation,size)
b_1 = np.zeros((n_1,1));
b_2 = np.zeros((n_2,1));

n_epoch = 30; #number of epochs
m = 10; #size of mini batch

def sigmoid(x):
	return 1/(1 + math.exp(-x));

def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x));	

sigmoid_v = np.vectorize(sigmoid);
sigmoid_prime_v = np.vectorize(sigmoid_prime);

for epoch in range(n_epoch):
	mini_batch_index = range(n_examples);
	random.shuffle(mini_batch_index); # mini_batch_index contains integers from 0 to 59,999 shuffled

	for i in range(1,n_examples/m):
		random_index = mini_batch_index[m*(i-1):m*i]; # taking integers from mini_batch_index in batches of m

		m_image_data = np.zeros((n_0,m)); # m examples randomly from the image data set
		y_data = np.zeros(m); # labels of the data

		for index in range(m):
			m_image_data[:,index] = images_norm[:,random_index[index]];
			y_data[index] = labels[random_index[index]];


		y = np.zeros((n_2,m));

		for index in range(m):
			y[int(y_data[index]),index] = 1.0

		a_0 = m_image_data;
		
		z_1 = np.dot(w_1,a_0) + np.dot(b_1,np.ones((1,m)));
		a_1 = sigmoid_v(z_1);

		z_2 = np.dot(w_2,a_1) + np.dot(b_2,np.ones((1,m)));
		a_2 = sigmoid_v(z_2);

		D_2 = a_2 - y;
		D_1 = np.multiply(np.dot(np.transpose(w_2),D_2),sigmoid_prime_v(z_1));
		
		w_2 = w_2 - (alpha/m)*np.dot(D_2,np.transpose(a_1));
		w_1 = w_1 - (alpha/m)*np.dot(D_1,np.transpose(a_0));

		b_2 = np.subtract(b_2,(alpha/m)*np.dot(D_2,np.ones((m,1))));
		b_1 = np.subtract(b_1,(alpha/m)*np.dot(D_1,np.ones((m,1))));

	# testing the accuracy on test data
	test_error = 0;

	for test_index in range(n_test_examples):
		test_image = images_test_norm[:,test_index];
		
		Y = labels_test[test_index];
		
		# doing a forward propagation on the test image
		A_0 = test_image;
		
		Z_1_temp = np.dot(w_1,A_0);
		Z_1_temp = np.reshape(Z_1_temp, (Z_1_temp.shape[0],1))
		Z_1 = Z_1_temp + b_1;
		
		A_1 = sigmoid_v(Z_1);

		Z_2_temp = np.dot(w_2,A_1);
		Z_2_temp = np.reshape(Z_2_temp, (Z_2_temp.shape[0],1))
		Z_2 = Z_2_temp + b_2;
		
		A_2 = sigmoid_v(Z_2);

		A_2_index = np.argmax(A_2); # number predicted by the NN

		if A_2_index != Y:
			test_error = test_error + 1;

	print "epoch number: ",epoch+1
	test_accuracy = 100.0 - (test_error)/100.0;

	print "test accuracy is: ", test_accuracy
	print "----------------------------------------"
	