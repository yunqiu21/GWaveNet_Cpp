all: gwavenet.cpp list.h nconv.h gcn.h adp.h nn/activation.h nn/batchnorm.h nn/conv.h nn/matmul.h nn/tensor.h
	g++ -std=c++14 gwavenet.cpp -o gwavenet

adp_test: adp_test.cpp adp.h nn/loader.h nn/tensor.h
	g++ -std=c++14 adp_test.cpp -o adp_test -ljsoncpp

clean: 
	rm gwavenet *.o