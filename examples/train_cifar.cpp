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
//    train_cifar.cpp:  train cifar-10 classifier
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
#include <tchar.h>

//#define MOJO_CV3
#include <mojo.h>  
#include <util.h>
#include "cifar_parser.h"

const int mini_batch_size = 16;
const float initial_learning_rate = 0.05f;
std::string solver = "adam";
std::string data_path = "../data/cifar-10-batches-bin/";
using namespace cifar;


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

void remove_cifar_mean(std::vector<std::vector<float>> &train_images, std::vector<std::vector<float>> &test_images)
{
	// calculate the mean for every pixel position 
	mojo::matrix mean(32, 32, 3);
	mean.fill(0);
	for (int i = 0; i < train_images.size(); i++) mean += mojo::matrix(32, 32, 3, train_images[i].data());
	mean *= (float)(1.f / train_images.size());

	// remove mean from data
	for (int i = 0; i < train_images.size(); i++)
	{
		mojo::matrix img(32, 32, 3, train_images[i].data());
		img -= mean;
		memcpy(train_images[i].data(), img.x, sizeof(float)*img.size());
	}
	for (int i = 0; i < test_images.size(); i++)
	{
		mojo::matrix img(32, 32, 3, test_images[i].data());
		img -= mean;
		memcpy(test_images[i].data(), img.x, sizeof(float)*img.size());
	}
}

int main()
{
	// == parse data
	// array to hold image data (note that mojo does not require use of std::vector)
	std::vector<std::vector<float>> test_images;
	std::vector<int> test_labels;
	std::vector<std::vector<float>> train_images;
	std::vector<int> train_labels;

	// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
	if (!parse_test_data(data_path, test_images, test_labels)) { std::cerr << "error: could not parse data.\n"; return 1; }
	if (!parse_train_data(data_path, train_images, train_labels)) { std::cerr << "error: could not parse data.\n"; return 1; }

	//remove_cifar_mean(train_images, test_images);

	// == setup the network  - when you train you must specify an optimizer ("sgd", "rmsprop", "adagrad", "adam")
	mojo::network cnn(solver.c_str());
	// !! the threading must be enabled with thread count prior to loading or creating a model !!
	cnn.enable_external_threads();
	cnn.set_mini_batch_size(mini_batch_size);
	cnn.set_smart_training(true); // automate training
	cnn.set_learning_rate(initial_learning_rate);
	// augment data random shifts only +/-2 pix
	cnn.set_random_augmentation(2,2,0,0,mojo::edge);

	// configure network 
	cnn.push_back("I1", "input 32 32 3");				// CIFAR is 32x32x3
	cnn.push_back("C1", "convolution 3 16 1 elu");		// 32-3+1=30
	cnn.push_back("P1", "semi_stochastic_pool 3 3");	// 10x10 out
	cnn.push_back("C2", "convolution 3 64 1 elu");		// 8x8 out
	cnn.push_back("P2", "semi_stochastic_pool 4 4");	// 2x2 out
	cnn.push_back("FC2", "softmax 10");
	
	// connect all the layers. Call connect() manually for all layer connections if you need more exotic networks.
	cnn.connect_all();
	std::cout << "==  Network Configuration  ====================================================" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;


	// add headers for table of values we want to log out
	mojo::html_log log;
	log.set_table_header("epoch\ttest accuracy(%)\testimated accuracy(%)\tepoch time(s)\ttotal time(s)\tlearn rate\tmodel");
	log.set_note(cnn.get_configuration());

	// setup timer/progress for overall training
	mojo::progress overall_progress(-1, "  overall:\t\t");
	const int train_samples = (int)train_images.size();
	while (1)
	{
		overall_progress.draw_header(data_name() + "  Epoch  " + std::to_string((long long)cnn.get_epoch() + 1), true);
		// setup timer / progress for this one epoch
		mojo::progress progress(train_samples, "  training:\t\t");

		cnn.start_epoch("cross_entropy");

		#pragma omp parallel for schedule(dynamic)
		for (int k = 0; k<train_samples; k++)
		{
			// augment data random shifts only
			//mojo::matrix m(32, 32, 3, train_images[k].data());
			//if (rand() % 2 == 0) m = m.flip_cols();
			//m = m.shift((rand() % 5) - 2, (rand() % 5) - 2, mojo::edge);
			cnn.train_class(train_images[k].data(), train_labels[k]);
			if (k % 1000 == 0) progress.draw_progress(k);
		}
		//		mojo::hide();
		cnn.end_epoch();
		//cnn.set_learning_rate(0.5f*cnn.get_learning_rate());
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

		// == run testing set
		progress.reset((int)test_images.size(), "  testing out-of-sample:\t");
		float accuracy = test(cnn, test_images, test_labels);
		std::cout << "  test accuracy:\t" << accuracy << "% (" << 100.f - accuracy << "% error)      " << std::endl;

		// save model
		std::string model_file = "../models/snapshots/tmp_" + std::to_string((long long)cnn.get_epoch()) + ".txt";
		cnn.write(model_file, true);
		std::cout << "  saved model:\t\t" << model_file << std::endl << std::endl;

		// write log file
		std::string log_out;
		log_out += float2str(dt) + "\t";
		log_out += float2str(overall_progress.elapsed_seconds()) + "\t";
		log_out += float2str(cnn.get_learning_rate()) + "\t";
		log_out += model_file;
		log.add_table_row(cnn.estimated_accuracy, accuracy, log_out);
		// will write this every epoch
		log.write("../models/snapshots/mojo_cifar_log.htm");

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

