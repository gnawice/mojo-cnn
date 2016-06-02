Pre-trained models for mojo cnn:
+ **mnist_deepcnet.mojo:** MNIST model 99.75% accuracy (0.25% error). Random +/-2 pixel translations on training data. No elastic distortions. Four convolution layers.  Each deepcnet layer is a 2x2 convolution followed by 2x2 max pool.  It took a little more than 2 hours to get to this accuracy. 
  ```  
input 28x28x1 identity  
convolution 3x3 40 elu
max_pool 2x2
deepcnet 80 elu
deepcnet 160 elu
deepcnet 320 elu
fully_connected 10 softmax
  ```  

+ **cifar_deepcnet.mojo:** CIFAR-10 model 87.55% accuracy (12.45% error) No mean subtraction. Random mirror and +/-2 pixel translations on training data. No rotation, scale, or elastic augmentation.  Five main convolution layers. Each deepcnet layer is a 2x2 convolution followed by 2x2 max pool.  It took a little more than 8.5 hours to get to this accuracy. 
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
