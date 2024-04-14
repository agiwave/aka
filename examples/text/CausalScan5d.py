import torch
import torch.cuda as cuda
import torch.utils.cpp_extension as ext
import os

script_dir = os.path.dirname(__file__)
if cuda.is_available():
    causal_scan_kernel = ext.load('causal_scan_5d', [
        os.path.join(script_dir, 'CausalScan5d.cu')
    ]) 
else:
    causal_scan_kernel = ext.load('causal_scan_5d', [
        os.path.join(script_dir, 'CausalScan5d.hpp')
    ]) 

class causal_scan(torch.autograd.Function):
    '''
    Formula:
    h(1) = a(1) * z         + b(1) * x(1)
    h(2) = a(2) * h(1)      + b(2) * x(2)
    ...
    h(n) = a(n) * h(n-1)    + b(n) * x(n)
    
    y(1) = c(1) * h(1)
    ...
    y(n) = c(n) * h(n)

    Return:
    (Y, h(n))
    '''
    @staticmethod
    def forward(ctx, H, A, B, X, C):
        (H, A, B, X, C) = [item.contiguous() for item in [H, A, B, X, C]]
        ZO = H.clone()
        O = torch.zeros_like(X)
        causal_scan_kernel.forward(ZO, A, B, X, C, O)
        ctx.save_for_backward(H, A, B, X, C)
        return O, ZO

    @staticmethod
    def backward(ctx, gradO, gradZO):
        H, A, B, X, C = ctx.saved_variables
        gradH, gradA, gradB, gradX, gradC = causal_scan_kernel.backward(gradO, H, A, B, X, C)
        return gradH, gradA, gradB, gradX, gradC

if __name__ == "__main__":
    device = torch.device("cuda")
    Z = torch.tensor([
        [[[1,1,1,1]]]
    ], device=device, dtype=torch.float)
    A = torch.tensor([
        [[[2]]],
        [[[2]]]
    ], device=device, dtype=torch.float)
    B = torch.tensor([
        [[[3,3,3,3]]],
        [[[3,3,3,3]]]
    ], device=device, dtype=torch.float)
    X = torch.tensor([
        [[[4]]],
        [[[4]]]
    ], device=device, dtype=torch.float)
    C = torch.tensor([
        [[[5,5,5,5]]],
        [[[5,5,5,5]]],
    ], device=device, dtype=torch.float)
    (Z, A, B, X, C) = [
       item.unsqueeze(0)
        for item in [Z, A, B, X, C]
    ]
    (A, B, X, C) = [
        torch.repeat_interleave(item, 2, dim=1)
        for item in [A, B, X, C]
    ]
    (Z, A, B, X) = [
        torch.repeat_interleave(item, 2, dim=2)
        for item in [Z, A, B, X]
    ]
    print(causal_scan.apply(Z, A, B, X, C))
