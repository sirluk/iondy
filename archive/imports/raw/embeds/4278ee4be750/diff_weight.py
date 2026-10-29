class DiffWeight(nn.Module):
    """
    Implementation of diff pruning weights using pytorch parametrizations
    https://pytorch.org/tutorials/intermediate/parametrizations.html
    """
    def __init__(
        self,
        weight: torch.Tensor,
        alpha_init: Union[float, int],
        concrete_lower: Union[float, int],
        concrete_upper: Union[float, int]
    ):
        super().__init__()
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        
        weight.requires_grad = False
        self.register_parameter("finetune", Parameter(torch.clone(weight)))
        self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))
        
    def forward(self, X):
        diff = (self.finetune - X).detach()
        return (self.finetune - diff) + self.z * (self.finetune - X)
    
    @property
    def z(self) -> Parameter:
        return self.dist(self.alpha)

    def dist(self, x) -> torch.Tensor:
        return concrete_stretched(
            x,
            l=self.concrete_lower,
            r=self.concrete_upper,
            deterministic=(not self.training)
        )