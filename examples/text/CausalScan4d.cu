#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __SHAPE_T__
#define __SHAPE_T__
typedef struct{
    int x, y, z, l, s;
}shape_t;
#endif//__SHAPE_T__

#ifndef IDX
#define IDX(shape) (((blockIdx.x % shape.x) * shape.l * shape.y + (blockIdx.y % shape.y) ) * shape.z + blockIdx.z % shape.z)
#endif//IDX

namespace { namespace device {
    template <typename scalar_t> __global__ void causalScan4d_Forward(
        int length,
        const shape_t shapeA,
        const shape_t shapeB,
        const shape_t shapeZ,
        scalar_t * Z,
        scalar_t * A,
        scalar_t * B,
        scalar_t * O
    )
    {
        int idx_A = IDX(shapeA);
        int idx_B = IDX(shapeB);
        int idx_Z = IDX(shapeZ);
        scalar_t zh = Z[idx_Z];
        while(length-->0) {
            zh = A[idx_A] * zh + B[idx_B];
            O[idx_B] = zh;
            idx_A += shapeA.s;
            idx_B += shapeB.s;
        }
    }

    template <typename scalar_t> __global__ void causalScan4d_Backward(
        int length,
        const shape_t shapeA,
        const shape_t shapeB,
        const shape_t shapeZ,
        scalar_t * gradZ,
        scalar_t * gradA,
        scalar_t * gradB,
        scalar_t * gradO,
        scalar_t * Z,
        scalar_t * A,
        scalar_t * O
    )
    {
        int idx_A = IDX(shapeA);
        int idx_B = IDX(shapeB);
        int idx_Z = IDX(shapeZ);
        idx_A += shapeA.s * (length - 1);
        idx_B += shapeA.s * (length - 1);
        scalar_t grad = 0.0;
        while(length-->1) {
            grad += gradO[idx_B];
            gradB[idx_B] = grad;
            atomicAdd(gradA + idx_A, O[idx_B - shapeB.s] * grad);
            grad *= A[idx_A];
            idx_A -= shapeA.s;
            idx_B -= shapeB.s;
        }
        grad += gradO[idx_B];
        gradB[idx_B] = grad;
        atomicAdd(gradA + idx_A, Z[idx_Z] * grad);
        gradZ[idx_Z] = A[idx_A] * grad;
    }
}}

#define __PYBINDED__
#include "./CausalScan4d.hpp"
torch::Tensor causalScan4d_cuda_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B) {
    if(!A.is_cuda()) {
        return causalScan4d_cpu_Forward(Z,A,B);
    }
    auto O = torch::zeros_like(B);
    int length = (int)O.size(1);
    shape_t shapeA = SHAPE4D(A);
    shape_t shapeB = SHAPE4D(B);
    shape_t shapeZ = SHAPE4D(Z);
    const dim3 blocks(O.size(0), O.size(2), O.size(3));
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan4d_Forward", ([&] {
        device::causalScan4d_Forward<scalar_t><<<blocks, 1>>>(
            length,
            shapeA,
            shapeB,
            shapeZ,
            (scalar_t*)Z.data_ptr(),
            (scalar_t*)A.data_ptr(),
            (scalar_t*)B.data_ptr(),
            (scalar_t*)O.data_ptr()
        );
    }));
    return O;
}

std::vector<torch::Tensor> causalScan4d_cuda_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O) {
    if(!A.is_cuda()) {
        return causalScan4d_cpu_Backward(gradO,Z,A,O);
    }
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(O.slice(1, 0, 1));
    int length = (int)O.size(1);
    shape_t shapeA = SHAPE4D(gradA);
    shape_t shapeB = SHAPE4D(gradB);
    shape_t shapeZ = SHAPE4D(gradZ);
    const dim3 blocks(O.size(0), O.size(2), O.size(3));
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan4d_Backward", ([&] {
        device::causalScan4d_Backward<scalar_t><<<blocks, 1>>>(
            length,
            shapeA,
            shapeB,
            shapeZ,
            (scalar_t*)gradZ.data_ptr(),
            (scalar_t*)gradA.data_ptr(),
            (scalar_t*)gradB.data_ptr(),
            (scalar_t*)gradO.data_ptr(),
            (scalar_t*)Z.data_ptr(),
            (scalar_t*)A.data_ptr(),
            (scalar_t*)O.data_ptr()
        );
    }));
    return {gradZ, gradA, gradB};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan4d_cuda_Forward, "");
    m.def("backward", &causalScan4d_cuda_Backward, "");
}