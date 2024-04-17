import torch

class ComplexExponential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.exp(input)  # Ensure this works on a batch
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * torch.exp(input)  # Element-wise multiplication
        return grad_input

