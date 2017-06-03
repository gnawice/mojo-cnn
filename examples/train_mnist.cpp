// == mojo ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files(the "Software"),
//    to deal in the Software without restriction, including without 
//    limitation the rights to use, copy, modify, merge, publish, distribute,
//    sublicense, and/or sell copies of the Software, and to permit persons to
//    whom the Software is furnished to do so, subject to the following 
//    conditions :
//
//    The above copyright notice and this permission notice shall be included
//    in all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
//    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ============================================================================
//    train_mnist.cpp:  train MNIST classifier
//
//    Instructions: 
//	  Add the "mojo" folder in your include path.
//    Download MNIST data and unzip locally on your machine:
//		(http://yann.lecun.com/exdb/mnist/index.html)
//    Set the data_path variable in the code to point to your data location.
// ==================================================================== mojo ==

#include <iostream> // cout
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
//#include <tchar.h>

//#define MOJO_CV3

#include <mojo.h>  
#include <util.h>
#include "mnist_parser.h"

const int mini_batch_size = 24;
const float initial_learning_rate = 0.04f;
std::string solver = "adam";
std::string data_path="../data/mnist/";
using namespace mnist;

// performs validation testing
float test(mojo::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels)
{
	// use progress object for simple timing and status updating
	mojo::progress progress((int)test_images.size(), "  testing:\t\t");

	int out_size = cnn.out_size(); // we know this to be 10 for MNIST
	int correct_predictions = 0;
	const int record_cnt = (int)test_images.size();

	#pragma omp parallel for reduction(+:correct_predictions) schedule(dynamic)
	for (int k = 0; k<record_cnt; k++)
	{
		const int prediction = cnn.predict_class(test_images[k].data());
		if (prediction == test_labels[k]) correct_predictions += 1;
		if (k % 1000 == 0) progress.draw_progress(k);
	}

	float accuracy = (float)correct_predictions / record_cnt*100.f;
	return accuracy;
}


int main()
{
	// ==== parse data
	// array to hold image data (note that mojo does not require use of std::vector)
	std::vector<std::vector<float>> test_images;
	std::vector<int> test_labels;
	std::vector<std::vector<float>> train_images;
	std::vector<int> train_labels;

	// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
	if (!parse_test_data(data_path, test_images, test_labels)) { std::cerr << "error: could not parse data.\n"; return 1; }
	if (!parse_train_data(data_path, train_images, train_labels)) { std::cerr << "error: could not parse data.\n"; return 1; }

	// ==== setup the network  - when you train you must specify an optimizer ("sgd", "rmsprop", "adagrad", "adam")
	mojo::network cnn(solver.c_str());
	// !! the threading must be enabled with thread count prior to loading or creating a model !!
	cnn.enable_external_threads();
	cnn.set_mini_batch_size(mini_batch_size);
	cnn.set_smart_training(true); // automate training
	cnn.set_learning_rate(initial_learning_rate);
	
	// Note, network descriptions can be read from a text file with similar format to the API
	cnn.read("../models/mnist_quickstart.txt");

	/*
	// to construct the model through API calls...
	cnn.push_back("I1", "input 28 28 1");				// MNIST is 28x28x1
	cnn.push_back("C1", "convolution 5 8 1 elu");		// 5x5 kernel, 20 maps. stride 1. out size is 28-5+1=24
	cnn.push_back("P1", "semi_stochastic_pool 3 3");	// pool 3x3 blocks. stride 3. outsize is 8
	cnn.push_back("C2i", "convolution 1 16 1 elu");		// 1x1 'inceptoin' layer
	cnn.push_back("C2", "convolution 5 48 1 elu");		// 5x5 kernel, 200 maps.  out size is 8-5+1=4
	cnn.push_back("P2", "semi_stochastic_pool 2 2");	// pool 2x2 blocks. stride 2. outsize is 2x2
	cnn.push_back("FC2", "softmax 10");					// 'flatten' of 2x2 input is inferred
	// connect all the layers. Call connect() manually for all layer connections if you need more exotic networks.
	cnn.connect_all();
	// */	

	std::cout << "==  Network Configuration  ====================================================" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;

	// add headers for table of values we want to log out
	mojo::html_log log;
	log.set_table_header("epoch\ttest accuracy(%)\testimated accuracy(%)\tepoch time(s)\ttotal time(s)\tlearn rate\tmodel");
	log.set_note(cnn.get_configuration());
				
	// augment data random shifts only
	cnn.set_random_augmentation(1,1,0,0,mojo::edge);

	// setup timer/progress for overall training
	mojo::progress overall_progress(-1, "  overall:\t\t");
	const int train_samples = (int)train_images.size();
	float old_accuracy = 0; 
	while (1)
	{
		overall_progress.draw_header(data_name() + "  Epoch  " + std::to_string((long long)cnn.get_epoch() + 1), true);
		// setup timer / progress for this one epoch
		mojo::progress progress(train_samples, "  training:\t\t");
		// set loss function
		cnn.start_epoch("cross_entropy");

		// manually loop through data. batches are handled internally. if data is to be shuffled, the must be performed externally
		#pragma omp parallel for schedule(dynamic)  // schedule dynamic to help make progress bar work correctly
		for (int k = 0; k<train_samples; k++)
		{
			cnn.train_class(train_images[k].data(), train_labels[k]);
			if (k % 1000 == 0) progress.draw_progress(k);
		}

		// draw weights of main convolution layers
		#ifdef MOJO_CV3
		mojo::show(mojo::draw_cnn_weights(cnn, "C1",mojo::tensorglow), 2 /* scale x 2 */, "C1 Weights");
		mojo::show(mojo::draw_cnn_weights(cnn, "C2",mojo::tensorglow), 2, "C2 Weights");
		#endif
		
		cnn.end_epoch();
		float dt = progress.elapsed_seconds();
		std::cout << "  mini batch:\t\t" << mini_batch_size << "                               " << std::endl;
		std::cout << "  training time:\t" << dt << " seconds on " << cnn.get_thread_count() << " threads" << std::endl;
		std::cout << "  model updates:\t" << cnn.train_updates << " (" << (int)(100.f*(1. - (float)cnn.train_skipped / cnn.train_samples)) << "% of records)" << std::endl;
		std::cout << "  estimated accuracy:\t" << cnn.estimated_accuracy << "%" << std::endl;


		/* if you want to run in-sample testing on the training set, include this code
		// == run training set
		progress.reset((int)train_images.size(), "  testing in-sample:\t");
		float train_accuracy=test(cnn, train_images, train_labels);
		std::cout << "  train accuracy:\t"<<train_accuracy<<"% ("<< 100.f - train_accuracy<<"% error)      "<<std::endl;
		*/

		// ==== run testing set
		progress.reset((int)test_images.size(), "  testing out-of-sample:\t");
		float accuracy = test(cnn, test_images, test_labels);
		std::cout << "  test accuracy:\t" << accuracy << "% (" << 100.f - accuracy << "% error)      " << std::endl;

		// if accuracy is improving, reset the training logic that may be thinking about quitting
		if (accuracy > old_accuracy)
		{
			cnn.reset_smart_training();
			old_accuracy = accuracy;
		}

		// save model
		std::string model_file = "../models/snapshots/tmp_" + std::to_string((long long)cnn.get_epoch()) + ".txt";
		cnn.write(model_file,true);
		std::cout << "  saved model:\t\t" << model_file << std::endl << std::endl;

		// write log file
		std::string log_out;
		log_out += float2str(dt) + "\t";
		log_out += float2str(overall_progress.elapsed_seconds()) + "\t";
		log_out += float2str(cnn.get_learning_rate()) + "\t";
		log_out += model_file;
		log.add_table_row(cnn.estimated_accuracy, accuracy, log_out);
		// will write this every epoch
		log.write("../models/snapshots/mojo_mnist_log.htm");

		// can't seem to improve
		if (cnn.elvis_left_the_building())
		{
			std::cout << "Elvis just left the building. No further improvement in training found.\nStopping.." << std::endl;
			break;
		}

	};
	std::cout << std::endl;
	return 0;
}