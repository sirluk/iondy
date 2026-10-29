def get_log_ratio(concrete_lower: float, concrete_upper: float) -> int:
    return 0 if (concrete_lower == 0) else math.log(-concrete_lower / concrete_upper)

def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
    return torch.sigmoid(alpha - log_ratio).sum()