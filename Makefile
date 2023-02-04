all: gwavenet.cpp list.h nconv.h gcn.h adp.h nn/activation.h nn/batchnorm.h nn/conv.h nn/matmul.h nn/tensor.h
	g++ -std=c++14 gwavenet.cpp -o gwavenet

adp_test: adp_test.cpp adp.h nn/loader.h nn/tensor.h utils/jsoncpp.o
	g++ -std=c++14 adp_test.cpp utils/jsoncpp.o -o adp_test

gcn_test: gcn_test.cpp gcn.h nn/loader.h nn/tensor.h list.h utils/jsoncpp.o
	g++ -std=c++14 gcn_test.cpp utils/jsoncpp.o -o gcn_test

gwavenet_test: gwavenet_test.cpp gwavenet.h gcn.h adp.h nn/conv.h nn/activation.h nn/loader.h nn/tensor.h nn/batchnorm.h list.h utils/jsoncpp.o
	g++ -std=c++14 -g gwavenet_test.cpp utils/jsoncpp.o -o gwavenet_test

jsoncpp.o: utils/jsoncpp.cpp utils/json/json.h utils/json/json-forwards.h
	g++ -std=c++14 -c utils/jsoncpp.cpp -o utils/jsoncpp.o

clean:
	rm -f gwavenet *.o *_test