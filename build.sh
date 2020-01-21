# g++ -std=c++11 -O2 -fPIC -shared -o libmatmul.so matmul.cpp -fopenmp
clang++ -std=c++11 -O2 -dynamiclib -shared -o libmatmul.so matmul.cpp
