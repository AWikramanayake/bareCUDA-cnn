#ifndef TCONT_H
#define TCONT_H

#include <stdexcept>

template<typename T>
class Tensor_cont {
    public:
    T* data;
    const int* shape;
    const int dims;
    const int length;

    Tensor_cont(int shapearr[], int numdims);
    Tensor_cont(const int*, const int);
    Tensor_cont(int);
    Tensor_cont(const Tensor_cont& a);
    Tensor_cont(Tensor_cont&& a);
    ~Tensor_cont();
    int lengthcalc(const int shapearr[], const int numdims);
    int* dimcopy(const Tensor_cont&);
    int* dimget(const int*, const int);
    void init_zeroes();
    void init_normaldist(float);
    void init_standardnormaldist(float);
    void prefetch2dvc(int);
    void prefetch2host(int);

    template<typename T>   
    Tensor_cont<T>::Tensor_cont& operator=(const Tensor_cont& a) {
    if (!((dims == a.dims) && (length == a.length))) {
        throw std::runtime_error("Tensor dimension mismatch in copy assignment operation\n");
    }
    for (int i = 0; i < dims; i++) {
        if (!(shape[i] == a.shape[i])) {
            throw std::runtime_error("Tensor dimension mismatch in copy assignment operation\n");
        }
    }

    T* p = new T[a.length];
    for (int i = 0; i < a.length; i++) {
        p[i] = a.data[i];
    }
    delete[] data;
    data = p;

    return *this;
    }

    template<typename T>   
    Tensor_cont& operator=(Tensor_cont&& a) {
        if (!((dims == a.dims) && (length == a.length))) {
            throw std::runtime_error("Tensor dimension mismatch in move assignment operation\n");
        }
        for (int i = 0; i < dims; i++) {
            if (!(shape[i] == a.shape[i])) {
                throw std::runtime_error("Tensor dimension mismatch in move assignment operation\n");
            }
        }
        data = a.data;
        a.data = nullptr;
        a.shape = nullptr;
    }

};

#endif