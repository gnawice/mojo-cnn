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
//    test.cpp:  Simple example using pre-trained model to test mojo cnn
//
//    Instructions: 
//	  Add the "mojo" folder in your include path.
//    Download MNIST data and unzip locally on your machine:
//		(http://yann.lecun.com/exdb/mnist/index.html)
//    Download CIFAR-10 data and unzip locally on your machine:
//		(http://www.cs.toronto.edu/~kriz/cifar.html)
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

//*
#include "mnist_parser.h"
using namespace mnist;
std::string data_path="../data/mnist/";
std::string model_file="../models/mojo_mnist.model";

/*/
#include "cifar_parser.h"
using namespace cifar;
std::string data_path="../data/cifar-10-batches-bin/";
std::string model_file="../models/mojo_cifar.model";
//*/

void test(mojo::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels)
{
	int out_size=cnn.out_size(); // we know this to be 10 for MNIST and CIFAR
	int correct_predictions=0;

	// use progress object for simple timing and status updating
	mojo::progress progress((int)test_images.size(), "  testing : ");

	const int record_cnt= (int)test_images.size();
	for(int k=0; k<record_cnt; k++)
	{
		// predict_class returnes the output index of the highest response
		const int prediction=cnn.predict_class(test_images[k].data());
		if(prediction ==test_labels[k]) correct_predictions++;
		if(k%1000==0) progress.draw_progress(k);
//#ifdef MOJO_CV3
//		mojo::show(mojo::draw_cnn_state(cnn,"C1",mojo::voodoo), 4, "State");
//#endif

	}
	float dt = progress.elapsed_seconds();
	std::cout << "  test time: " << dt << " seconds                                          "<< std::endl;
	std::cout << "  records: " << test_images.size() << std::endl;
	std::cout << "  speed: " << (float)record_cnt/dt << " records/second" << std::endl;
	std::cout << "  accuracy: " << (float)correct_predictions/record_cnt*100.f <<"%" << std::endl;
}


int main()
{
	// == parse data
	// array to hold image data (note that mojo does not require use of std::vector)
	std::vector<std::vector<float>> test_images;
	// array to hold image labels 
	std::vector<int> test_labels;
	// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
	if(!parse_test_data(data_path, test_images, test_labels)) {std::cerr << "error: could not parse data.\n"; return 1;}

	// == setup the network  
	mojo::network cnn; 
	// load model
	if(!cnn.read(model_file)) {std::cerr << "error: could not read model.\n"; return 1;}
	std::cout << "Mojo CNN Configuration:" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;

//#ifdef MOJO_CV3
//	mojo::show(mojo::draw_cnn_weights(cnn, mojo::mojo_palette::gray,1), 7, "Weights");
//#endif

	// == run the test
	std::cout << "Testing " << data_name() << ":" << std::endl;
	// this function will loop through all images, call predict, and print out stats
	test(cnn, test_images, test_labels);	
	std::cout << std::endl;
	return 0;
}
