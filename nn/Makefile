tensor_test: tensor_test.cpp tensor.h
	g++ -std=c++14 tensor_test.cpp -o tensor_test

conv_test: conv_test.cpp conv.h tensor.h
	g++ -std=c++14 conv_test.cpp jsoncpp.o -o conv_test

loader_test: loader_test.o jsoncpp.o loader.h tensor.h
	g++ -std=c++14 -o loader_test loader_test.o jsoncpp.o

loader_test.o: loader_test.cpp loader.h tensor.h ../utils/json/json.h
	g++ -std=c++14 -c loader_test.cpp

jsoncpp.o: ../utils/jsoncpp.cpp ../utils/json/json.h ../utils/json/json-forwards.h
	g++ -std=c++14 -c ../utils/jsoncpp.cpp

batchnorm_test: batchnorm_test.cpp batchnorm.h loader.h tensor.h
	g++ -std=c++14 batchnorm_test.cpp jsoncpp.o -o batchnorm_test

activation_test: activation_test.cpp activation.h tensor.h
	g++ -std=c++14 activation_test.cpp -o activation_test

matmul_test: matmul_test.cpp matmul.h tensor.h
	g++ -std=c++14 matmul_test.cpp -o matmul_test

clean:
	rm *_test *.o