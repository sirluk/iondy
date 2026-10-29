class DiffWeightFixmask(nn.Module):
    
    def __init__(self, pre_trained: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.register_parameter("pre_trained", Parameter(pre_trained, requires_grad=False))
        self.register_parameter("mask", Parameter(mask, requires_grad=False))
    
    def forward(self, X):
        return self.pre_trained + self.mask * X   