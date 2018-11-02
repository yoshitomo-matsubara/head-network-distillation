from utils import module_util


def wrap_all_child_modules(model, wrapper_class, member_name=None, member_module=None, wrapped_list=list(), **kwargs):
    named_children = model.named_children() if member_module is None else member_module.named_children()
    named_children = list(named_children)
    if not named_children and member_name is not None and member_module is not None:
        wrapped_module = wrapper_class(member_module, **kwargs)
        setattr(model, member_name, wrapped_module)
        wrapped_list.append(wrapped_module)
        return

    parent = model if member_module is None else member_module
    for name, child_module in named_children:
        wrap_all_child_modules(parent, wrapper_class, name, child_module, wrapped_list, **kwargs)


def wrap_decomposable_modules(middle_module, wrapper_class, z, middle_name=None, upper_module=None,
                              wrapped_list=list(), first=True, **kwargs):
    named_children = list(middle_module.named_children())
    if not named_children and upper_module is not None:
        try:
            return middle_module(z), True
        except (RuntimeError, ValueError):
            try:
                return middle_module(z.view(z.size(0), -1)), True
            except RuntimeError:
                print('Error\t', type(middle_module).__name__)
        return z, False

    try:
        expected_z = middle_module(z)
    except (RuntimeError, ValueError):
        return z, False

    for name, child_module in named_children:
        z, flag = wrap_decomposable_modules(child_module, wrapper_class, z, name,
                                            middle_module, wrapped_list, False, **kwargs)

    named_children = list(middle_module.named_children())
    if flag and expected_z.size() == z.size() and expected_z.isclose(z).all().item() == 1:
        for name, child_module in named_children:
            no_wrap = True
            for m in child_module.children():
                if isinstance(m, wrapper_class):
                    no_wrap = False
                    break

            if not no_wrap or isinstance(child_module, wrapper_class):
                continue

            wrapped_module = wrapper_class(child_module, **kwargs)
            setattr(middle_module, name, wrapped_module)
    elif upper_module is not None:
        no_wrap = True
        for m in middle_module.children():
            if isinstance(m, wrapper_class):
                no_wrap = False
                break

        if no_wrap:
            wrapped_module = wrapper_class(middle_module, **kwargs)
            setattr(upper_module, middle_name, wrapped_module)

    if first:
        module_util.extract_target_modules(middle_module, wrapper_class, wrapped_list)
    return expected_z, True
