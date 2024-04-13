#include <torch/extension.h>
#include <vector>

torch::Tensor causalScan4d_cuda_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B);
std::vector<torch::Tensor> causalScan4d_cuda_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan4d_cuda_Forward, "");
    m.def("backward", &causalScan4d_cuda_Backward, "");
}