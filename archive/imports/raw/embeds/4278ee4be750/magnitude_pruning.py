def _get_cutoff(values, pct):
    k = int(len(values) * pct)
    return torch.topk(torch.abs(values), k, largest=True, sorted=True)[0][-1]            

diff_weights = torch.tensor([])
for base_module in get_encoder_base_modules():
    for n, par_list in list(base_module.parametrizations.items()):
        par_list[0].set_mode(train=False)
        diff_weight = (getattr(base_module, n) - par_list.original).detach().cpu()
        diff_weights = torch.cat([diff_weights, diff_weight.flatten()])
cutoff = _get_cutoff(diff_weights, pct)
for base_module in get_encoder_base_modules():
    for n, par_list in list(base_module.parametrizations.items()):
        pre_trained_weight = torch.clone(par_list.original)
        parametrize.remove_parametrizations(base_module, n)
        p = base_module._parameters[n].detach()
        diff_weight = (p - pre_trained_weight)
        diff_mask = (torch.abs(diff_weight) >= cutoff)
        base_module._parameters[n] = Parameter(diff_weight * diff_mask)
        parametrize.register_parametrization(
            base_module,
            n,
            DiffWeightFixmask(pre_trained_weight, diff_mask)
        )