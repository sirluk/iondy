sparsity_pen = 1.25e-7

loss = loss_fn(outputs, labels.to(self.device))

l0_pen = 0.
for module_name, base_module in get_encoder_base_modules(return_names=True):
    for par_list in list(base_module.parametrizations.values()):
        l0_norm = get_l0_norm_term(par_list[0].alpha, log_ratio)
        l0_pen += (l0_norm * sparsity_pen)
loss += l0_pen