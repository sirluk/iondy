class AdvHead(nn.Module):
    
    def __init__(self, adv_count: int = 1, **kwargs):
        super().__init__()    
        self.heads = nn.ModuleList()
        for i in range(adv_count):
            self.heads.append(ClfHead(**kwargs))
            
    def forward(self, x):
        out = []
        for head in self.heads:
            out.append(head(x))
        return out
            
    def forward_reverse(self, x, lmbda = 1.):
        x_ = ReverseLayerF.apply(x, lmbda)
        return self(x_)
    
    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()