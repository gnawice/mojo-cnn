CC=g++
CFLAGS= -I../mojo/ -std=c++11 -O3 -DMOJO_SSE3 -msse4 -mavx 
CFLAGS_OMP= -I../mojo/ -std=c++11 -fopenmp -O3 -DMOJO_OPM -DMOJO_SSE3 -msse4 -mavx

all: test test_omp train_mnist train_cifar

test: test.cpp 
	$(CC) $(CFLAGS) test.cpp -o test

test_omp: test_omp.cpp
	$(CC) $(CFLAGS_OMP) test_omp.cpp -o test_omp

train_mnist: train_mnist.cpp
	$(CC) $(CFLAGS_OMP) train_mnist.cpp -o train_mnist

train_cifar: train_cifar.cpp 
	$(CC) $(CFLAGS_OMP) train_cifar.cpp -o train_cifar

clean:
	-rm -f test
	-rm -f test_omp
	-rm -f train_mnist
	-rm -f train_cifar
