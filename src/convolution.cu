#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/tensor_container.cu"
#include <cuda_fp16.h>

#define OUTDIM 24

__global__ void convolve(Tensor_cont<__half2>& output, Tensor_cont<__half>& image, Tensor_cont<float>& filter, Tensor_cont<__half>& bias, int in_dim, int out_dim) {
    __shared__ __half2 imgslice[143];
    __shared__ __half2 outslice[32];
    __shared__ __half2 filt[200];
    
    int tid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
    //islice_x/y are the offset of the thread from imgstartpoint
    int islice_x = blockDim.x*threadIdx.y + threadIdx.x;
    int islice_y = threadIdx.z;
    int imgstartpoint = in_dim*blockIdx.y*blockDim.z + blockIdx.x*blockDim.y;

    
    if (islice_x < 11) {
        for (int i = islice_y; i < 13; i += blockDim.z) {
                imgslice[i*11 + islice_x] = *((half2*)&image.data[imgstartpoint + in_dim*i + islice_x]);
        }
    }

    for (int idx = tid; idx < 200; idx += blockDim.x*blockDim.y+blockDim.z) {
        filt[tid] = __float2half2_rn(filter.data[blockIdx.z*25 + tid]);
    }

    __half2 placeholder = __float2half2_rn(0.0);

    for (int i = 0; i < filter.shape[3]; i++) {
        placeholder = __hfma2(filt[i*5 + threadIdx.x], imgslice[(i+threadIdx.z)*11 + 2*blockIdx.y + blockIdx.x], placeholder);
    }

    if (threadIdx.x == 0) {
        outslice[threadIdx.z*4 + threadIdx.y] = __hfma2_relu(outslice[threadIdx.z*4 + threadIdx.y], __half2(0.0,0.0), __half2(bias.data[threadIdx.z], bias.data[threadIdx.z]));
    }

    atomicAdd(&outslice[threadIdx.z*4 + threadIdx.y], placeholder);
    __syncthreads();

    if (threadIdx.x == 0) {
        /* 
            Terms in output.data[] index:
            1. Offset to get to correct filter copy of img
            2. Offset to get to correct y in 3x3 subdivision on image
            3. Offset to get to correct x in 3x3 subdivision
            4. Offset to get to correct y within 3x3 subdivision
            5. Offset to get to correct x within 3x3 subdivision 
        */
        // No race conditions here. Only 1 thread should be assigned to each entry in output.
        output.data[out_dim*out_dim*blockIdx.z + out_dim*blockDim.z*blockIdx.y + blockIdx.x*blockDim.y + out_dim*threadIdx.z + threadIdx.y] = outslice[threadIdx.z*4 + threadIdx.y];
    }

}

Tensor_cont<__half2> convolution (Tensor_cont<__half2> const image, Tensor_cont<__half2> const filter, Tensor_cont<__half2> const bias, int in_dim, int s=1) {  

}


/*
int main() {
}
*/