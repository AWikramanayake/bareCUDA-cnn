#include "utils/mnist_parser.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>

typedef unsigned char uchar;

__global__ void print_3(uchar* images, uchar* labels, int img_size, int num_data) {
        for (int i = 0, j = 0; i < img_size*3; i++) {
        if (i%28 == 0) {
            printf("\n");
        }
        if (i%784 == 0) {
            printf("\n");
            printf("%d\n", labels[j]);
            j++;
        }
        printf("%3u ", images[i]);
    }
    printf("\n\n");
    int offset1 = 50000*img_size;
    int offset2 = 50000;

    for (int i = offset1, j = offset2; i < offset1 + img_size*3; i++) {
        if (i%28 == 0) {
            printf("\n");
        }
        if (i%784 == 0) {
            printf("\n");
            printf("%d\n", labels[j]);
            j++;
        }
        printf("%3u ", images[i]);
    }
    printf("%d %d", num_data, img_size);
}

int main(int argc, char* args[]) {
    /*
        DATA EXTRACTION
        Creates a pointer to a 60000*784 array of uchars (X) and a 60000 array of uchars (Y)
        The arrays are shuffled using the seed
    */
    int num_data = 60000;
    
    thrust::host_vector<int> permutation(num_data, 0);
    thrust::sequence(permutation.begin(), permutation.end());
    int seed = 130432;
    thrust::shuffle(thrust::host, permutation.begin(), permutation.end(), thrust::default_random_engine(seed));

    int img_size;
    std::string img_path = "train-images.idx3-ubyte";
    std::string label_path = "train-labels.idx1-ubyte";
    auto images = read_mnist_images(img_path, num_data, img_size, permutation);
    auto labels = read_mnist_labels(label_path, num_data, permutation);

    /*
        GPU transfer
        Move X and Y to the GPU
    */

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    cudaMemPrefetchAsync(images, sizeof(unsigned char)*num_data*img_size, deviceId);
    cudaMemPrefetchAsync(labels, sizeof(unsigned char)*num_data, deviceId);

    printf("starting kernel call\n");
    cudaDeviceSynchronize();
    print_3<<<1,1>>>(images, labels, img_size, num_data);
    cudaDeviceSynchronize();

}