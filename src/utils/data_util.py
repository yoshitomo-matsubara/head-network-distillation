def convert2type_list(str_var, delimiter, var_type):
    return list(map(var_type, str_var.split(delimiter)))


def convert2type_range(str_var, delimiter, var_type):
    return range(*convert2type_list(str_var, delimiter, var_type))
