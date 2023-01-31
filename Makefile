all: gwavenet.cpp list.h nconv.h gcn.h adp.h nn/activation.h nn/batchnorm.h nn/conv.h nn/matmul.h nn/tensor.h
	g++ -std=c++14 gwavenet.cpp -o gwavenet

clean: 
	rm gwavenet *.o