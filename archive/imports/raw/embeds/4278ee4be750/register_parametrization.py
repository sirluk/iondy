parametrize.register_parametrization(
    module_name,
    param_name,
    DiffWeight(param, alpha_init, conc_upper, conc_lower)
)