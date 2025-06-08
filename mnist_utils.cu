#include <stdio.h>
#include <assert.h>
#include <string>
#include "cooperative_groups.h"
#include <cmath>
#include "cooperative_groups/reduce.h"
#include <iostream>
#include <random>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>

namespace cg = cooperative_groups;

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

/*
 * Calculates the sum of an array of size n at location pointed by *data, and places the sum at *sum 
*/
__global__ void reduce(float* sum, float* data, int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    float v = 0;

    for (int tid = grid.thread_rank(); tid < n; tid += grid.size()) {
        v += data[tid];
    }
    warp.sync();
    v = cg::reduce(warp, v, cg::plus<float>());

    if (warp.thread_rank() == 0) {
        atomicAdd(sum, v);
    }
}


/*
 * Calculates the deviation (NOT the standard deviation!) of values in array at *data of size n from the value 'mean'
 * and places the deviation at *std
*/
__global__ void calc_deviation(float* std, float* data, int n, float mean) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    float v = 0;
    float temp = 0;

    for (int tid = grid.thread_rank(); tid < n; tid += grid.size()) {
        temp = mean;
        temp -= data[tid];
        v += temp*temp;
    }

    warp.sync();
    v = cg::reduce(warp, v, cg::plus<float>());

    if (warp.thread_rank() == 0) {
        atomicAdd(std, v);
    }
}


/*
 * Divides the values in an array at *data by 255, subtracts the mean from each value and divides by the standard deviation
 * The result should be a normalized array with mean = 0 and stddev = 1
*/
__global__ void normalize_mnist(float* data, float mean, float stddev, int n, int image_size) {
    auto grid = cg::this_grid();
    for (int tid = grid.thread_rank(); tid < n*image_size; tid += grid.size()) {
        data[tid] /= 255.0;
        data[tid] -= mean;
        data[tid] /= stddev;
    }
}


/*
 * Reads n labels from a valid mnist label file at 'full_path'. The labels are written to an array at *data in the order given by 'permutation'
 * 'read_number_of_labels' is used to store the number of labels read from the file
*/
void read_mnist_labels(int* data, int n, int read_number_of_labels, std::string full_path, thrust::host_vector<int> permutation) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char*)&read_number_of_labels, sizeof(read_number_of_labels)), read_number_of_labels = reverseInt(read_number_of_labels);

        if (n <= read_number_of_labels) {
            printf("Reading %d of %d labels\n", n, read_number_of_labels);
        } else {
            throw std::runtime_error("Attempted to read more labels than contained in file `" + full_path + "`!");
        }

        for(int i = 0; i < n; i++) {
            file.read((char*)&data[permutation[i]], 1);
        }

    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}


/*
 * Reads n images of size 'image_size' from a valid mnist image file at 'full_path'. The images are written to an array at *data in the order given by 'permutation'
 * 'read_number_of_images' is used to store the number of labels read from the file
*/
void read_mnist_images(float* data, int n, int& read_number_of_images, int& image_size, std::string full_path, thrust::host_vector<int> permutation) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&read_number_of_images, sizeof(read_number_of_images)), read_number_of_images = reverseInt(read_number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        if (n <= read_number_of_images) {
            printf("Reading %d of %d labels\n", n, read_number_of_images);
        } else {
            throw std::runtime_error("Attempted to read more labels than contained in file `" + full_path + "`!");
        }

        image_size = n_rows * n_cols; 

        unsigned char temp = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < image_size; j++) {
                file.read((char*)&temp, 1);
                data[permutation[i]*image_size + j] = temp;
            }
        }
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}



// example usage
/*
int main() {
    int* a;
    float* b;
    int n = 100;
    int img_size = 28*28;
    int size = n*img_size;
    int read_n_img;
    int read_n_label;
    std::string label_path = "train-labels.idx1-ubyte";
    std::string img_path = "train-images.idx3-ubyte";

    thrust::host_vector<int> permutation(n, 0);
    thrust::sequence(permutation.begin(), permutation.end());
    int seed = 130452;
    thrust::shuffle(thrust::host, permutation.begin(), permutation.end(), thrust::default_random_engine(seed));

    checkCuda(cudaMallocManaged(&a, n*sizeof(int)));
    checkCuda(cudaMallocManaged(&b, n*img_size*sizeof(float)));

    read_mnist_labels(a, n, read_n_label,  label_path, permutation);
    read_mnist_images(b, n, read_n_img, img_size, img_path, permutation);


    for (int i = 0, j = 0; i < img_size*2; i++) {
        if (i%28 == 0) {
            printf("\n");
        }
        if (i%784 == 0) {
            printf("\n");
            printf("%d\n", a[j]);
            j++;
        }
        printf("%.1f ", b[i]);
    }
    printf("\n\n");

    
    int blockSize = 256;
    int nBlocks = (n + blockSize - 1) / blockSize;

    float *sum, *dev;
    cudaMallocManaged(&sum, sizeof(float));
    cudaMallocManaged(&dev, sizeof(float));

    cudaMemset(sum, 0, sizeof(float));
    cudaMemset(dev, 0, sizeof(float));
    cudaDeviceSynchronize();
    reduce<<<nBlocks, blockSize>>>(sum, b, size);
    cudaDeviceSynchronize();

    float mean = (*(sum))/size;

    calc_deviation<<<nBlocks, blockSize>>>(dev, b, size, mean);
    cudaDeviceSynchronize();

    float stddev = sqrt(((*dev) / size)); 
    
    printf("mean = %f\nstddev = %f\n", mean, stddev);
    printf("\n");

    mean /= 255.0;
    stddev /= 255.0;

    printf("normalized mean = %f\nnormalized stddev = %f\n", mean, stddev);
    printf("\n");

    normalize_mnist<<<nBlocks, blockSize>>>(b, mean, stddev, n, img_size);
    cudaDeviceSynchronize();

    cudaMemset(sum, 0, sizeof(float));
    cudaMemset(dev, 0, sizeof(float));
    cudaDeviceSynchronize();
    reduce<<<nBlocks, blockSize>>>(sum, b, size);
    cudaDeviceSynchronize();

    mean = (*(sum))/size;

    calc_deviation<<<nBlocks, blockSize>>>(dev, b, size, mean);
    cudaDeviceSynchronize();

    stddev = sqrt(((*dev) / size)); 

    printf("new mean = %f\nnew stddev = %f\n", mean, stddev);
    printf("\n\n");

    for (int i = 0, j = 0; i < img_size*2; i++) {
        if (i%28 == 0) {
            printf("\n");
        }
        if (i%784 == 0) {
            printf("\n");
            printf("%d\n", a[j]);
            j++;
        }
        printf("%.1f ", b[i]);
    }
    printf("\n");

    cudaFree(sum);
    cudaFree(dev);
    cudaFree(a);
    cudaFree(b);
}
*/