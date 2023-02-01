tensor_test: tensor_test.cpp tensor.h
	g++ -std=c++14 tensor_test.cpp -o tensor_test

conv_test: conv_test.cpp tensor.h
	g++ -std=c++14 conv_test.cpp -o conv_test

loader_test: loader_test.cpp loader.h tensor.h
	g++ -std=c++14 loader_test.cpp -o loader_test -ljsoncpp

batchnorm_test: batchnorm_test.cpp batchnorm.h tensor.h
	g++ -std=c++14 batchnorm_test.cpp -o batchnorm_test -ljsoncpp

activation_test: activation_test.cpp activation.h module.h tensor.h
	g++ -std=c++14 activation_test.cpp -o activation_test

clean: 
	rm *_test