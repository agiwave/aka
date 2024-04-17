#ifndef __COMMON_H__
#define __COMMON_H__
typedef struct{
    int b, l, d, stepb, stepl;
}shape_t;
typedef struct{
    int x;
}INDICS;
#define SHAPE4D(t) {(int)t.size(0), (int)t.size(1), (int)t.size(2), \
    (int)(t.size(1)*t.size(2)), \
    (t.size(1)==1)?0:(int)t.size(2) \
}
#define SHIFT_BLOCK_SIZE 8
#define BLOCK_SIZE (1<<SHIFT_BLOCK_SIZE)
#define IDX_SCALE(shape) ((ib % shape.b) * shape.stepb + id % shape.d)
#define IDX(shape) (ib * shape.stepb + id)
#endif//__COMMON_H__

#ifndef __DISABLE_CUDA__
    #define DEVICEINDICS 
    #define CAUSAL_FORWARD causalScan_Forward_cuda
    #define CAUSAL_BACKWARD causalScan_Backward_cuda
    #define atomAdd atomicAdd
    #include <cuda.h>
    #include <cuda_runtime.h>
#else//__DISABLE_CUDA__
    #ifdef DEVICEINDICS
        #undef DEVICEINDICS
    #endif//
    #ifdef CAUSAL_FORWARD
        #undef CAUSAL_FORWARD
    #endif//
    #ifdef CAUSAL_BACKWARD
        #undef CAUSAL_BACKWARD
    #endif//
    #ifdef __global__
        #undef __global__
    #endif//
    #ifdef atomAdd
        #undef atomAdd
    #endif//
    #define DEVICEINDICS ,const INDICS& blockIdx, const INDICS& threadIdx
    #define CAUSAL_FORWARD causalScan_Forward_cpu
    #define CAUSAL_BACKWARD causalScan_Backward_cpu
    #ifdef __global__
        #undef __global__
    #endif//
    #define __global__
    #define atomAdd(p,b) (*(p) = *(p) + (b))
#endif//__DISABLE_CUDA__

namespace { namespace device {
    template <typename scalar_t> __global__ void CAUSAL_FORWARD(
        const shape_t shapeA,
        const shape_t shapeO,
        const shape_t shapeZ,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pX,
        scalar_t * pO,
        int range
        DEVICEINDICS
    )
    {
        int idx = blockIdx.x << SHIFT_BLOCK_SIZE | threadIdx.x;
        if( idx >= range ) return;
        int ib = idx / shapeO.d;
        int id = idx % shapeO.d;

        int idxX = IDX(shapeO);
        pA += IDX_SCALE(shapeA);
        pX += idxX;
        pO += idxX;
        scalar_t zh = pZ[IDX(shapeZ)];
        int length = shapeO.l;
        while(length-->0) {
            zh = (*pA) * zh + (*pX);
            (*pO) = zh;
            pA += shapeA.stepl;
            pX += shapeO.stepl;
            pO += shapeO.stepl;
        }
    }

    template <typename scalar_t> __global__ void CAUSAL_BACKWARD(
        const shape_t shapeA,
        const shape_t shapeO,
        const shape_t shapeZ,
        scalar_t * gradZ,
        scalar_t * gradA,
        scalar_t * gradX,
        scalar_t * gradO,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pO,
        int range
        DEVICEINDICS
    )
    {
        int idx = blockIdx.x << SHIFT_BLOCK_SIZE | threadIdx.x;
        if( idx >= range ) return;
        int ib = idx / shapeO.d;
        int id = idx % shapeO.d;
        
        scalar_t grad = 0.0;
        int idxA = IDX_SCALE(shapeA);
        int idxO = IDX(shapeO);
        int idxZ = IDX(shapeZ);
        int length = shapeO.l;
        int offsetA = idxA + shapeA.stepl * (length - 1);
        int offsetO = idxO + shapeO.stepl * (length - 1);
        pA += offsetA;
        pO += offsetO-shapeO.stepl; // h(n-1)
        gradA += offsetA;
        gradX += offsetO;
        gradO += offsetO;
        while(length-->1) {
            grad += *gradO;
            (*gradX) = grad;
            atomAdd(gradA, (*pO) * grad);
            grad *= (*pA);
            gradA -= shapeA.stepl;
            gradX -= shapeO.stepl;
            gradO -= shapeO.stepl;
            pA -= shapeA.stepl;
            pO -= shapeO.stepl;
        }
        grad += *gradO;
        (*gradX) = grad;
        atomAdd(gradA, pZ[idxZ] * grad);
        gradZ[idxZ] = (*pA) * grad;
    }
}}

#ifndef __TORCH_INLINE__
#define __TORCH_INLINE__

#ifndef __DISABLE_CUDA__
#define __DISABLE_CUDA__
#include "CausalScan.cu"
#undef __DISABLE_CUDA__
#endif//__DISABLE_CUDA__

#include <torch/extension.h>
#include <vector>
torch::Tensor causalScan_Forward(torch::Tensor X, torch::Tensor Z, torch::Tensor A) {
    auto O = torch::empty_like(X);
    shape_t shapeA = SHAPE4D(A);
    shape_t shapeO = SHAPE4D(X);
    shape_t shapeZ = SHAPE4D(Z);
    int range = (int)(shapeO.b * shapeO.d);
    if(A.is_cuda()) {
        #ifndef __DISABLE_CUDA__
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Forward", ([&] {
            int blocks = (range + BLOCK_SIZE - 1) >> SHIFT_BLOCK_SIZE;
            device::causalScan_Forward_cuda<scalar_t><<<blocks, BLOCK_SIZE>>>(
                shapeA,
                shapeO,
                shapeZ,
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)X.data_ptr(),
                (scalar_t*)O.data_ptr(),
                range
            );
        }));
        #else//
        AT_ASSERT(false);
        #endif//__DISABLE_CUDA__
    }else{
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Forward", ([&] {
            at::parallel_for(0, range, 0, [&](int64_t start, int64_t end){
                while(start<end){
                    INDICS indics[] = {
                        {(int)(start >> SHIFT_BLOCK_SIZE)},
                        {(int)(start % BLOCK_SIZE)}
                    };
                    device::causalScan_Forward_cpu<scalar_t>(
                        shapeA,
                        shapeO,
                        shapeZ,
                        (scalar_t*)Z.data_ptr(),
                        (scalar_t*)A.data_ptr(),
                        (scalar_t*)X.data_ptr(),
                        (scalar_t*)O.data_ptr(),
                        range,
                        indics[0],
                        indics[1]
                    );
                    start++;
                }
            });
        }));
    }
    return O;
}

std::vector<torch::Tensor> causalScan_Backward(torch::Tensor gradO, torch::Tensor O, torch::Tensor Z, torch::Tensor A) {
    auto gradA = torch::zeros_like(A);
    auto gradX = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(Z);
    shape_t shapeA = SHAPE4D(gradA);
    shape_t shapeO = SHAPE4D(gradX);
    shape_t shapeZ = SHAPE4D(gradZ);
    int range = (int)(shapeO.b * shapeO.d);
    if(A.is_cuda()) {
        #ifndef __DISABLE_CUDA__
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Backward", ([&] {
            int blocks = (range + BLOCK_SIZE - 1) >> SHIFT_BLOCK_SIZE;
            device::causalScan_Backward_cuda<scalar_t><<<blocks, BLOCK_SIZE>>>(
                shapeA,
                shapeO,
                shapeZ,
                (scalar_t*)gradZ.data_ptr(),
                (scalar_t*)gradA.data_ptr(),
                (scalar_t*)gradX.data_ptr(),
                (scalar_t*)gradO.data_ptr(),
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)O.data_ptr(),
                range
            );
        }));
        #else//
        AT_ASSERT(false);
        #endif//__DISABLE_CUDA__
    }
    else{
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Backward", ([&] {
            for(int start=0; start<range; start++){
                INDICS indics[] = {
                    {(int)(start >> SHIFT_BLOCK_SIZE)},
                    {(int)(start % BLOCK_SIZE)}
                };
                device::causalScan_Backward_cpu(
                    shapeA,
                    shapeO,
                    shapeZ,
                    (scalar_t*)gradZ.data_ptr(),
                    (scalar_t*)gradA.data_ptr(),
                    (scalar_t*)gradX.data_ptr(),
                    (scalar_t*)gradO.data_ptr(),
                    (scalar_t*)Z.data_ptr(),
                    (scalar_t*)A.data_ptr(),
                    (scalar_t*)O.data_ptr(),
                    range,
                    indics[0],
                    indics[1]
                );
            }
            // at::parallel_for(0, shapeO.b * shapeZ.d, 0, [&](int64_t start, int64_t end){
            //     while(start<end){
            //         INDICS indics[] = {
            //             {(int)(start/shapeZ.d), (int)(start%shapeZ.d)}
            //         };
            //         start++;
            //     }
            // });
        }));
    }

    return {gradX, gradZ, gradA};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan_Forward, "");
    m.def("backward", &causalScan_Backward, "");
}
#endif//__TORCH_INLINE__