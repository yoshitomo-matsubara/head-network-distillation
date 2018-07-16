import os


def get_file_list(dir_path, is_recursive=False, is_sorted=False):
    file_list = list()
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        if os.path.isfile(path):
            file_list.append(path)
        elif is_recursive:
            file_list.extend(get_file_list(path, is_recursive))
    return sorted(file_list) if is_sorted else file_list


def get_dir_list(dir_path, is_recursive=False, is_sorted=False):
    dir_list = list()
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        if os.path.isdir(path):
            dir_list.append(path)
        elif is_recursive:
            dir_list.extend(get_dir_list(path, is_recursive))
    return sorted(dir_list) if is_sorted else dir_list


def make_dirs(dir_path):
    if len(dir_path) > 0 and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def make_parent_dirs(file_path):
    dir_path = os.path.dirname(file_path)
    make_dirs(dir_path)
