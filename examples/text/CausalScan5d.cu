#include <cuda.h>
#include <cuda_runtime.h>
#ifndef __wrap_t__
#define __wrap_t__
template <typename scalar_t> struct wrap_t{
    int x, y, z, l, n, s;
    scalar_t* p;
};
#endif//__wrap_t__

#ifndef IDX5D
#define IDX5D(shape) ((((blockIdx.x % shape.x) * shape.l * shape.y + blockIdx.y % shape.y) * shape.z + blockIdx.z % shape.z) * shape.n + threadIdx.x % shape.n)
#define Ptr5D(shape) (shape.p + ((((blockIdx.x % shape.x) * shape.l * shape.y + blockIdx.y % shape.y ) * shape.z + blockIdx.z % shape.z) * shape.n + threadIdx.x % shape.n))
#endif//IDX5D

#define atomAdd atomicAdd

namespace { namespace device {
    template <typename scalar_t> __global__ void causalScan5d_Forward(
        const wrap_t<scalar_t> shapeZ,
        const wrap_t<scalar_t> shapeA,
        const wrap_t<scalar_t> shapeB,
        const wrap_t<scalar_t> shapeX,
        const wrap_t<scalar_t> shapeC,
        const wrap_t<scalar_t> shapeO
    )
    {
        scalar_t * pZ = Ptr5D(shapeZ);
        scalar_t * pA = Ptr5D(shapeA);
        scalar_t * pB = Ptr5D(shapeB);
        scalar_t * pX = Ptr5D(shapeX);
        scalar_t * pC = Ptr5D(shapeC);
        scalar_t * pO = Ptr5D(shapeO);
        int length = shapeO.l;
        scalar_t zh = *pZ;
        while(length-->0) {
            zh = (*pA) * zh + (*pB) * (*pX);
            atomAdd(pO, ((*pC) * zh));
            pA += shapeA.s;
            pB += shapeB.s;
            pX += shapeX.s;
            pC += shapeC.s;
            pO += shapeO.s;
        }
        *pZ = zh;
    }

    template <typename scalar_t> __global__ void causalScan5d_Backward(
        const wrap_t<scalar_t> shapeZ,
        const wrap_t<scalar_t> shapeA,
        const wrap_t<scalar_t> shapeB,
        const wrap_t<scalar_t> shapeX,
        const wrap_t<scalar_t> shapeC,
        const wrap_t<scalar_t> gradO,
        scalar_t * gradZ,
        scalar_t * gradA,
        scalar_t * gradB,
        scalar_t * gradX,
        scalar_t * gradC
    )
    {
        int length = gradO.l;
        scalar_t * pZ = Ptr5D(shapeZ);
        scalar_t * pA = Ptr5D(shapeA);
        scalar_t * pB = Ptr5D(shapeB);
        scalar_t * pX = Ptr5D(shapeX);
        scalar_t * pC = Ptr5D(shapeC) + shapeC.s * length;
        scalar_t * pGradO = Ptr5D(gradO) + gradO.s * length;;
        scalar_t * pGradZ = gradZ;
        scalar_t * pGradA = gradA + shapeA.s * length;
        scalar_t * pGradB = gradB + shapeB.s * length;
        scalar_t * pGradX = gradX + shapeX.s * length;
        scalar_t * pGradC = gradC + shapeC.s * length;

        scalar_t * zhs = new scalar_t[length+1];
        zhs[0] = *pZ;
        for(int i=0; i<length; i++) {
            zhs[i+1] = (*pA) * zhs[i] + (*pB) * (*pX);
            pA += shapeA.s;
            pB += shapeB.s;
            pX += shapeX.s;
        }

        scalar_t grad = 0.0;
        while(length-->0) {
            pA -= shapeA.s;
            pB -= shapeB.s;
            pX -= shapeX.s;
            pC -= shapeC.s;
            pGradO -= gradO.s;
            pGradB -= shapeB.s;
            pGradX -= shapeX.s;
            pGradC -= shapeC.s;

            atomAdd(pGradC, (*pGradO) * zhs[length+1]);
            grad += (*pGradO) * (*pC);
            atomAdd(pGradB, grad * (*pX));
            atomAdd(pGradX, grad * (*pB));
            atomAdd(pGradA, zhs[length] * grad);
            grad *= (*pA);
        }
        *pGradZ = grad;
        delete[] zhs;
    }
}}

#undef atomAdd
#define __PYBINDED__
#include "./CausalScan5d.hpp"
torch::Tensor causalScan5d_cuda_Forward(
    torch::Tensor Z, 
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor X, 
    torch::Tensor C,
    torch::Tensor O
) {
    if(!O.is_cuda()) {
        return causalScan5d_cpu_Forward(Z, A, B, X, C, O);
    }

    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan5d_Forward", ([&] {
        wrap_t<scalar_t> shapeZ = SHAPE5D(Z);
        wrap_t<scalar_t> shapeA = SHAPE5D(A);
        wrap_t<scalar_t> shapeB = SHAPE5D(B);
        wrap_t<scalar_t> shapeX = SHAPE5D(X);
        wrap_t<scalar_t> shapeC = SHAPE5D(C);
        wrap_t<scalar_t> shapeO = SHAPE5D(O);
        int threads = shapeZ.n;
        const dim3 blocks(O.size(0), O.size(2), O.size(3));    
        device::causalScan5d_Forward<scalar_t><<<blocks, threads>>>(
            shapeZ,
            shapeA,
            shapeB,
            shapeX,
            shapeC,
            shapeO
        );
    }));
    return O;
}

std::vector<torch::Tensor> causalScan5d_cuda_Backward(
    torch::Tensor gradO,
    torch::Tensor Z,
    torch::Tensor A,
    torch::Tensor B, torch::Tensor X, 
    torch::Tensor C
) {
    if(!gradO.is_cuda()) {
        return causalScan5d_cpu_Backward(gradO, Z, A, B, X, C);
    }
    auto gradZ = torch::zeros_like(Z);
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(B);
    auto gradX = torch::zeros_like(X);
    auto gradC = torch::zeros_like(C);
    AT_DISPATCH_FLOATING_TYPES(gradO.type(), "causalScan5d_Backward", ([&] {
        wrap_t<scalar_t> shapeZ = SHAPE5D(Z);
        wrap_t<scalar_t> shapeA = SHAPE5D(A);
        wrap_t<scalar_t> shapeB = SHAPE5D(B);
        wrap_t<scalar_t> shapeX = SHAPE5D(X);
        wrap_t<scalar_t> shapeC = SHAPE5D(C);
        wrap_t<scalar_t> shapeGradO = SHAPE5D(gradO);
        int threads = shapeZ.n;
        const dim3 blocks(gradO.size(0), gradO.size(2), gradO.size(3));
        device::causalScan5d_Backward<scalar_t><<<blocks, threads>>>(
            shapeZ,
            shapeA,
            shapeB,
            shapeX,
            shapeC,
            shapeGradO,
            (scalar_t*)gradZ.data_ptr(),
            (scalar_t*)gradA.data_ptr(),
            (scalar_t*)gradB.data_ptr(),
            (scalar_t*)gradX.data_ptr(),
            (scalar_t*)gradC.data_ptr()
        );
    }));
    return {gradZ, gradA, gradB, gradX, gradC};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan5d_cuda_Forward, "");
    m.def("backward", &causalScan5d_cuda_Backward, "");
}