#include <cuda.h>
#include <cuda_runtime.h>

typedef struct{
    int b;
    int l;
    int h;
    int d;
}shape_t;

namespace { namespace device {
    template <typename scalar_t> __global__ void causalScan4d_Forward(
        const shape_t shapeA,
        scalar_t * Z,
        scalar_t * A,
        scalar_t * B,
        scalar_t * O
    )
    {
        int idx_A = ((blockIdx.x % shapeA.b) * shapeA.l * shapeA.h + blockIdx.y) * shapeA.d + blockIdx.z % shapeA.d;
        int idx_B = (blockIdx.x * shapeA.l * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
        int idx_Z = (blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
        int step_A = shapeA.h * shapeA.d;
        int step_B = gridDim.y * gridDim.z;
        scalar_t zh = Z[idx_Z];
        int length = shapeA.l;
        while(length-->0) {
            zh = A[idx_A] * zh + B[idx_B];
            O[idx_B] = zh;
            idx_A += step_A;
            idx_B += step_B;
        }
    }

    template <typename scalar_t> __global__ void causalScan4d_Backward(
        const shape_t shapeA,
        scalar_t * gradZ,
        scalar_t * gradA,
        scalar_t * gradB,
        scalar_t * gradO,
        scalar_t * Z,
        scalar_t * A,
        scalar_t * O
    )
    {
        int idx_A = ((blockIdx.x % shapeA.b) * shapeA.l * shapeA.h + blockIdx.y) * shapeA.d + blockIdx.z % shapeA.d;
        int idx_B = (blockIdx.x * shapeA.l * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
        int idx_Z = (blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
        int step_A = shapeA.h * shapeA.d;
        int step_B = gridDim.y * gridDim.z;
        int length = shapeA.l;
        idx_A += step_A * (length - 1);
        idx_B += step_B * (length - 1);
        scalar_t grad = 0.0;
        while(length-->1) {
            grad += gradO[idx_B];
            gradB[idx_B] = grad;
            atomicAdd(gradA + idx_A, O[idx_B - step_B] * grad);
            grad *= A[idx_A];
            idx_A -= step_A;
            idx_B -= step_B;
        }
        grad += gradO[idx_B];
        gradB[idx_B] = grad;
        atomicAdd(gradA + idx_A, Z[idx_Z] * grad);
        gradZ[idx_Z] = A[idx_A] * grad;
    }
}}

#include <torch/extension.h>
#include <vector>

torch::Tensor causalScan4d_cpu_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B);
torch::Tensor causalScan4d_cuda_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B) {
    if(!A.is_cuda()) {
        return causalScan4d_cpu_Forward(Z,A,B);
    }
    auto O = torch::zeros_like(B);
    shape_t shapeA = {(int)A.size(0), (int)A.size(1), (int)A.size(2), (int)A.size(3)};
    const dim3 blocks(O.size(0), O.size(2), O.size(3));
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan4d_Forward", ([&] {
        device::causalScan4d_Forward<scalar_t><<<blocks, 1>>>(
            shapeA,
            (scalar_t*)Z.data_ptr(),
            (scalar_t*)A.data_ptr(),
            (scalar_t*)B.data_ptr(),
            (scalar_t*)O.data_ptr()
        );
    }));
    return O;
}

std::vector<torch::Tensor> causalScan4d_cpu_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O);
std::vector<torch::Tensor> causalScan4d_cuda_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O) {
    if(!A.is_cuda()) {
        return causalScan4d_cpu_Backward(gradO,Z,A,O);
    }
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(O.slice(1, 0, 1));
    shape_t shapeA = {(int)A.size(0), (int)A.size(1), (int)A.size(2), (int)A.size(3)};
    const dim3 blocks(O.size(0), O.size(2), O.size(3));
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan4d_Backward", ([&] {
        device::causalScan4d_Backward<scalar_t><<<blocks, 1>>>(
            shapeA,
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
