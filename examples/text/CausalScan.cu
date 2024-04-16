#ifndef __COMMON_H__
#define __COMMON_H__
typedef struct{
    int b, l, d, stepb, stepl;
}shape_t;
typedef struct{
    int x, y;
}INDICS;
#define SHAPE4D(t) {(int)t.size(0), (int)t.size(1), (int)t.size(2), \
    (int)(t.size(1)*t.size(2)), \
    (t.size(1)==1)?0:(int)t.size(2) \
}
#define IDX_SCALE(shape) ((blockIdx.x % shape.b) * shape.stepb + blockIdx.y % shape.d)
#define IDX(shape) (blockIdx.x * shape.stepb + blockIdx.y)
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
    #define DEVICEINDICS ,const INDICS& blockIdx
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
        scalar_t * pO
        DEVICEINDICS
    )
    {
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
        scalar_t * pO
        DEVICEINDICS
    )
    {
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
torch::Tensor causalScan_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B) {
    auto O = torch::zeros_like(B);
    shape_t shapeA = SHAPE4D(A);
    shape_t shapeO = SHAPE4D(B);
    shape_t shapeZ = SHAPE4D(Z);
    if(A.is_cuda()) {
        #ifndef __DISABLE_CUDA__
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Forward", ([&] {
            const dim3 blocks(O.size(0), O.size(2));
            device::causalScan_Forward_cuda<scalar_t><<<blocks, 1>>>(
                shapeA,
                shapeO,
                shapeZ,
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)B.data_ptr(),
                (scalar_t*)O.data_ptr()
            );
        }));
        #else//
        AT_ASSERT(false);
        #endif//__DISABLE_CUDA__
    }else{
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Forward", ([&] {
            at::parallel_for(0, shapeO.b * shapeZ.d, 0, [&](int64_t start, int64_t end){
                while(start<end){
                    INDICS indics[] = {
                        {(int)(start/shapeZ.d), (int)(start%shapeZ.d)}
                    };
                    device::causalScan_Forward_cpu<scalar_t>(
                        shapeA,
                        shapeO,
                        shapeZ,
                        (scalar_t*)Z.data_ptr(),
                        (scalar_t*)A.data_ptr(),
                        (scalar_t*)B.data_ptr(),
                        (scalar_t*)O.data_ptr(),
                        indics[0]
                    );
                    start++;
                }
            });
        }));
    }
    return O;
}

std::vector<torch::Tensor> causalScan_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O) {
    auto gradA = torch::zeros_like(A);
    auto gradX = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(Z);
    shape_t shapeA = SHAPE4D(gradA);
    shape_t shapeO = SHAPE4D(gradX);
    shape_t shapeZ = SHAPE4D(gradZ);
    if(A.is_cuda()) {
        #ifndef __DISABLE_CUDA__
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Backward", ([&] {
            const dim3 blocks(O.size(0), O.size(2));
            device::causalScan_Backward_cuda<scalar_t><<<blocks, 1>>>(
                shapeA,
                shapeO,
                shapeZ,
                (scalar_t*)gradZ.data_ptr(),
                (scalar_t*)gradA.data_ptr(),
                (scalar_t*)gradX.data_ptr(),
                (scalar_t*)gradO.data_ptr(),
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)O.data_ptr()
            );
        }));
        #else//
        AT_ASSERT(false);
        #endif//__DISABLE_CUDA__
    }
    else{
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Backward", ([&] {
            at::parallel_for(0, shapeO.b * shapeZ.d, 0, [&](int64_t start, int64_t end){
                while(start<end){
                    INDICS indics[] = {
                        {(int)(start/shapeZ.d), (int)(start%shapeZ.d)}
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
                        indics[0]
                    );
                    start++;
                }
            });
        }));
    }

    return {gradZ, gradA, gradX};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan_Forward, "");
    m.def("backward", &causalScan_Backward, "");
}
#endif//__TORCH_INLINE__