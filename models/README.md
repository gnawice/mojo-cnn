Pre-trained models for mojo cnn:
+ **mojo_mnist.model:** MNIST model 99.57% accuracy (0.43% error). Random +/-2 pixel translations on training data. No elastic distortions. Two convolution layers. Little over 28 min to get to this accuracy.
  ```  
input 28x28x1 identity  
convolution 5x5 20 elu  
dropout 0.20  
semi_stochastic_pool 4 4  
convolution 5x5 200 elu  
semi_stochastic_pool 2 2  
fully_connected 100 identity  
dropout 0.40  
fully_connected 10 tanh
  ```  

+ **mojo_cifar.model:** CIFAR-10 model 81.08% accuracy (18.9% error) No mean subtraction. Random mirror and +/-2 pixel translations on training data. No rotation or scale augmentation.  Three convolution layers. When subtracting mean, you can get to the same answer a little faster (2 hr instead of 2 hr 10 min)
  ```
input 32x32x3 identity  
convolution 5x5 32 elu  
semi_stochastic_pool 2 2  
convolution 5x5 32 elu  
semi_stochastic_pool 2 2  
convolution 5x5 64 elu  
fully_connected 100 identity  
dropout 0.25  
fully_connected 10 tanh  
  ```
