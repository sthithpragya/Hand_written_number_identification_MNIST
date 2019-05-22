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
n_layers = 2; # layers except the initial layer

alpha = 0.05;

#initialising weight and bias matrices/ vectors
neuron_dict = {}; # dictionary to store number of elements in each layer
neuron_dict["0"] = 784;
neuron_dict["1"] = 30;
neuron_dict["2"] = 10;

size_dict = {}; # dictionary to store size tuples of various weight matrices
size_dict["1"] = (30,784);
size_dict["2"] = (10,30);

w_dict = {}; # dictionary to save weight arrays
for index in range(n_layers):
	w_dict[str(index+1)] = np.random.normal(0,1/np.sqrt(neuron_dict[str(index)]),size_dict[str(index+1)]);

b_dict = {}; # store the bias vectors
for index in range(n_layers):
	b_dict[str(index+1)] = np.zeros((neuron_dict[str(index+1)],1));

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

		m_image_data = np.zeros((neuron_dict["0"],m)); # m examples randomly from the image data set
		y_data = np.zeros(m); # labels of the data

		for index in range(m):
			m_image_data[:,index] = images_norm[:,random_index[index]];
			y_data[index] = labels[random_index[index]];

		temp_key = neuron_dict.keys()[-1];
		y = np.zeros((neuron_dict[str(temp_key)],m));

		for index in range(m):
			y[int(y_data[index]),index] = 1.0

		a_dict = {}; # activation layer
		z_dict = {}; # pre-activation layer
		D_dict = {}; # error dictionary

		a_dict["0"] = m_image_data;

		for j in range(n_layers):

			z_dict[str(j+1)] = np.dot(w_dict[str(j+1)],a_dict[str(j)]) + np.dot(b_dict[str(j+1)],np.ones((1,m)));
			a_dict[str(j+1)] = sigmoid_v(z_dict[str(j+1)]);
			
		for j in range(n_layers,0,-1):
			if j == n_layers:
				D_dict[str(j)] = a_dict[str(j)] - y;
			else:
				D_dict[str(j)] = np.multiply(np.dot(np.transpose(w_dict[str(j+1)]),D_dict[str(j+1)]),sigmoid_prime_v(z_dict[str(j)]));

			w_dict[str(j)] = w_dict[str(j)] - (alpha/m)*np.dot(D_dict[str(j)],np.transpose(a_dict[str(j-1)]));
			b_dict[str(j)] = np.subtract(b_dict[str(j)],(alpha/m)*np.dot(D_dict[str(j)],np.ones((m,1))));

	# testing the accuracy on test data
	test_error = 0;

	for test_index in range(n_test_examples):
		test_image = images_test_norm[:,test_index];
		
		Y = labels_test[test_index];
		
		# doing a forward propagation on the test image
		A_dict = {};
		Z_dict = {};

		A_dict["0"] = test_image;
		
		for j in range(n_layers):
			Z_temp = np.dot(w_dict[str(j+1)],A_dict[str(j)]);
			Z_temp = np.reshape(Z_temp, (Z_temp.shape[0],1))			
			Z_dict[str(j+1)] = Z_temp + b_dict[str(j+1)];
			
			A_dict[str(j+1)] = sigmoid_v(Z_dict[str(j+1)]);
			
		last_key = A_dict.keys()[-1];
		A_last = A_dict[str(last_key)];
		A_last_index = np.argmax(A_last);

		if A_last_index != Y:
			test_error = test_error + 1;

	print "epoch number: ",epoch+1
	test_accuracy = 100.0 - (test_error)/100.0;

	print "test accuracy is: ", test_accuracy
	print "----------------------------------------"
	