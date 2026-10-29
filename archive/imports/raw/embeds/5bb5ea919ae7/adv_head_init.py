  def __init__(self, adv_count: int = 1, **kwargs):
      super().__init__()    
      self.heads = nn.ModuleList()
      for i in range(adv_count):
          self.heads.append(ClfHead(**kwargs))