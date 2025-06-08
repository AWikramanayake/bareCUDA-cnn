#ifndef MPARSE_H
#define MPARSE_H

#include <string>
#include <thrust/host_vector.h>

__global__ void reduce(float* sum, float* data, int n);
__global__ void calc_deviation(float* std, float* data, int n, float mean);
__global__ void normalize_mnist(float* data, float mean, float stddev, int n, int image_size);
void read_mnist_labels(int* data, int n, int read_number_of_labels, std::string full_path, thrust::host_vector<int> permutation);
void read_mnist_images(float* data, int n, int& read_number_of_images, int image_size, std::string full_path, thrust::host_vector<int> permutation);


#endif