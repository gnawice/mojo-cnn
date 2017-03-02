Pre-trained models for mojo cnn:

+ **vgg16.mojo:** VGG 16 layer model converted from mxnet model.  This model is a pretrained model on ILSVRC2012 dataset. It is able to achieve 71.0% Top-1 Accuracy and 89.8% Top-5 accuracy on ILSVRC2012-Validation Set. Details about the network architecture can be found in the following arXiv paper:
  ```
Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556
  ```
Please cite the paper if you use the model.    
[**Download vgg16.mojo**](https://drive.google.com/file/d/0B5Dx9ePCIXQAZU51T0MyQXpvOXc/view?usp=sharing)

+ **mnist_deepcnet.mojo:** MNIST model 99.75% accuracy (0.25% error). Random +/-2 pixel translations on training data. No elastic distortions. Four convolution layers.  Each deepcnet layer is a 2x2 convolution followed by 2x2 max pool.  It took a little more than 2 hours to get to this accuracy in original mojo release. 
  ```  
input 28x28x1 identity  
convolution 3x3 40 elu
max_pool 2x2
deepcnet 80 elu
deepcnet 160 elu
deepcnet 320 elu
fully_connected 10 softmax
  ```  

+ **cifar_deepcnet.mojo:** CIFAR-10 model 87.55% accuracy (12.45% error) No mean subtraction. Random mirror and +/-2 pixel translations on training data. No rotation, scale, or elastic augmentation.  Five main convolution layers. Each deepcnet layer is a 2x2 convolution followed by 2x2 max pool.  It took a little more than 8.5 hours to get to this accuracy in original mojo release. 
  ```
input 32x32x3 identity  
convolution 3x3 50 elu
max_pool 2x2
deepcnet 100 elu
deepcnet 150 elu
resize 7 7
deepcnet 200 elu
deepcnet 250 elu
fully_connected 10 tanh
  ```
