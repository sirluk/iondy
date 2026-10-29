  def _get_mean_loss(self, outputs: Union[torch.Tensor, List[torch.Tensor]], labels: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
      if isinstance(outputs, torch.Tensor):
          outputs = [outputs]
      losses = []
      for output in outputs:
          losses.append(loss_fn(output, labels))
      return torch.stack(losses).mean()    