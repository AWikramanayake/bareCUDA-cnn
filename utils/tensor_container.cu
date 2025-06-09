#include "tensor_container.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>


template<typename T>
int Tensor_cont<T>::lengthcalc(const int shapearr[], const int numdims) {
    int _length = 1;
    for (int i = 0; i < numdims; i++) {
        _length *= shapearr[i];
    }
    return _length;
}


template<typename T>
int* Tensor_cont<T>::dimcopy(const Tensor_cont& inp) {
    auto _output = new int[inp.dims];
    for (int i = 0; i < inp.dims; i++) {
        _output[i] = inp.shape[i];
    }
    return _output;
}


template<typename T>
int* Tensor_cont<T>::dimget(const int* shape, const int dims) {
    auto _output = new int[dims];
    for (int i = 0; i < dims; i++) {
        _output[i] = shape[i];
    }
    return _output;
}


template<typename T>
Tensor_cont<T>::Tensor_cont(int shapearr[], int numdims) :
dims(numdims),
shape(shapearr),
length(lengthcalc(shapearr, numdims))
{
    //data = new T[length];
    T* ptr;
    cudaMallocManaged(&ptr, (length * sizeof(T)));
    cudaDeviceSynchronize();
    this->data = ptr;
}


template<typename T>
Tensor_cont<T>::Tensor_cont(int shape) :
dims(1),
shape(shape),
length(shape)
{
    //data = new T[length];
    T* ptr;
    cudaMallocManaged(&ptr, (length * sizeof(T)));
    cudaDeviceSynchronize();
    this->data = ptr;
}


template<typename T>
Tensor_cont<T>::Tensor_cont(const int* shapearrptr, const int numdims) :
dims(numdims),
shape(dimget(shapearrptr, numdims)),
length(lengthcalc(shape, numdims))
{
    //data = new T[length];
    T* ptr;
    cudaMallocManaged(&ptr, (length * sizeof(T)));
    cudaDeviceSynchronize();
    this->data = ptr;
}


template<typename T>    
Tensor_cont<T>::~Tensor_cont() {
    //delete[] data;
    cudaDeviceSynchronize();
    cudaFree(this->data);
}


template<typename T>   
Tensor_cont<T>::Tensor_cont(const Tensor_cont& a) :
dims(a.dims),
shape(dimcopy(a)),
length(lengthcalc(shape, dims))            
{
    for (int i = 0; i < length; i++) {
        //data = new T[length];
        T* ptr;
        cudaMallocManaged(&ptr, (length * sizeof(T)));
        cudaDeviceSynchronize();
        this->data = ptr;
        this->data[i] = a.data[i];
        cudaDeviceSynchronize();
    }
}


template<typename T>   
Tensor_cont<T>::Tensor_cont(Tensor_cont&& a) :
dims(a.dims),
data(a.data),
shape(dimcopy(a)),
length(a.length) 
{
    a.data = nullptr;
    a.shape = nullptr;
}


void Tensor_cont<float>::init_zeroes() {
    for (int i = 0; i < length; i++) {
        data[i] = 0.0;
    }
}


void Tensor_cont<__half>::init_zeroes() {
    for (int i = 0; i < length; i++) {
        data[i] = 0.0;
    }
}


void Tensor_cont<__half2>::init_zeroes() {
    for (int i = 0; i < length; i++) {
        data[i] = {0.0, 0.0};
    }
}


void Tensor_cont<float>::init_normaldist(float mean) {
    float _stddev = 1.0/sqrt(length);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, _stddev);

    for (int i = 0; i < length; i++) {
        data[i] = distribution(generator);
    }
}


void Tensor_cont<__half>::init_normaldist(float mean) {
    float _stddev = 1.0/sqrt(length);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, _stddev);

    for (int i = 0; i < length; i++) {
        data[i] = __half(distribution(generator));
    }
}


void Tensor_cont<float>::init_standardnormaldist(float mean) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < length; i++) {
        data[i] = distribution(generator);
    }
}


void Tensor_cont<__half>::init_standardnormaldist(float mean) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < length; i++) {
        data[i] = __half(distribution(generator));
    }
}


template<typename T>
void Tensor_cont<T>::prefetch2dvc(int deviceID) {
    cudaMemPrefetchAsync(this->data, (length * sizeof(T)), deviceID);
}


template<typename T>
void Tensor_cont<T>::prefetch2host(int hostID) {
    cudaMemPrefetchAsync(this->data, (length * sizeof(T)), cudaCpuDeviceId);
}


// Explicit instantiation
template class Tensor_cont<float>;
template class Tensor_cont<__half>;
