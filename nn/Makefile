tensor_test: tensor_test.cpp tensor.h
	g++ -std=c++14 tensor_test.cpp -o tensor_test

conv_test: conv_test.cpp tensor.h module.h
	g++ -std=c++14 conv_test.cpp -o conv_test

loader_test: loader_test.cpp loader.h tensor.h
	g++ -std=c++14 loader_test.cpp -o loader_test -ljsoncpp

clean: 
	rm *_test