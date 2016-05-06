# mojo cnn (c++ convolutional neural network)

Mojo is an efficient C++ CNN implementation that was built with the goal to balance hack-ability, functionality, and speed.  Consisting of only a handful of header files, mojo is in portable C++ with old fashioned C tricks for optimization. Built with OpenMP and SSE3, it's speed is competitive with other CPU based CNN frameworks. Being a minimal CPU solution, it is not designed to scale over a cluster to train very deep models (for that, go with GPUs and Caffe, TensorFlow, CMTK, Torch, etc…)

The mojo cnn API provides a 'smart training' option which abstracts the management of the training process but still provides the flexibility to handle the threading and input data as you'd like. Just make a loop and pass in training samples until mojo cnn says stop. On the standard MNIST handwritten digit database, mojo's 'smart training' gives 99% accuracy in less than a minute. 

Latest change status is on the [mojo cnn wiki](https://github.com/gnawice/mojo-cnn/wiki). 

Features:
+ Layers:  Input, Fully Connected, Convolution, Max Pool, Semi-Stochastic Pool, Dropout, (Fractional Max Pool, Maxout-like pooling, Concatenation all in progress). [Read more on the wiki](https://github.com/gnawice/mojo-cnn/wiki/Layers)
+ Activation Functions: Identity, Hyperbolic Tangent (tanh), Exponential Linear Unit (ELU), Rectified Linear Unit (ReLU), Leaky Rectified Linear Unit (LReLU), Very Leaky Rectified Linear Unitv (VLReLU), Sigmoid, (Softmax in progress)
+ Optimization: Stochastic Gradient Descent, RMSProp, AdaGrad, Adam
+ Loss Functions: Mean Squared Error, Cross Entropy
+ Threading: optional and externally controlled at the application level using OpenMP
+ Architecture: Branching allowed
+ Solver: Smart training optimizes parameters, speeds up training, and provides exit criteria.
+ Image Support: optional OpenCV utilities (in progress)
+ Portable: tested with MS Developer Studio 2010, 2015, and Cygwin g++ 5.3.0. 
+ Logging: html training report

API Examples:
Load model and perform prediction:
```
#include <mojo.h>

mojo::network cnn; 
cnn.read("../models/mojo_mnist.model");
const int predicted_class=cnn.predict_class(float_image.data());

```

Construction of a new CNN for MNIST, and train records with OpenMP threading:  
```
#define MOJO_OMP
#include <omp.h>

ucnn::network cnn("adam");
cnn.set_smart_train(true);
cnn.enable_omp();
cnn.set_mini_batch_size(24);
	
// add layer definitions	
cnn.push_back("I1","input 28 28 1");            // MNIST is 28x28x1
cnn.push_back("C1","convolution 5 5 15 elu");   // 5x5 kernel, 12 maps.  out size is 28-5+1=24
cnn.push_back("P1","semi_stochastic_pool 4 4"); // pool 4x4 blocks, stride 4. out size is 6
cnn.push_back("C2","convolution 5 5 150 elu");  // 5x5 kernel, 150 maps.  out size is 6-5+1=2
cnn.push_back("P2","semi_stochastic_pool 2 2"); // pool 2x2 blocks. out size is 2/2=1 
cnn.push_back("FC1","fully_connected 100 identity");// fully connected 100 nodes 
cnn.push_back("FC2","fully_connected 10 tanh"); 
 
cnn.connect_all(); // connect layers automatically (no branches)

while(1)
{
	// train with OpenMP threading
	cnn.start_epoch("cross_entropy");
	
	#pragma omp parallel
	#pragma omp for schedule(dynamic)
	for(int k=0; k<train_samples; k++) cnn.train_class(train_images[k].data(), train_labels[k]);
	
	cnn.end_epoch();
	
	std::cout << "estimated accuracy:" << cnn.estimated_accuracy << "%" << std::endl;
	
	cnn.write("mojo_tmp.model");
	
	if (cnn.elvis_left_the_building()) break;
};

```

Example training log from sample application:
![](https://github.com/DozerTheCat/ucnn/wiki/images/log_example.jpg)


