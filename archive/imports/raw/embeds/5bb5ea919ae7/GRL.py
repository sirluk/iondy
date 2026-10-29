from torch.autograd import Function

class ReverseLayerF(Function):
    # https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/2
    @staticmethod
    def forward(ctx, input_, lmbda):
        ctx.lmbda = lmbda
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lmbda
        return grad_input, None