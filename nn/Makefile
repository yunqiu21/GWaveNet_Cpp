tensor_test: tensor_test.cpp tensor.h
	g++ -std=c++14 tensor_test.cpp -o tensor_test

conv_test: conv_test.cpp tensor.h module.h
	g++ -std=c++14 conv_test.cpp -o conv_test

activation_test: activation_test.cpp activation.h module.h tensor.h
	g++ -std=c++14 activation_test.cpp -o activation_test

clean: 
	rm *_test