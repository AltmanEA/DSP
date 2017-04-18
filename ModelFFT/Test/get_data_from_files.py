import string


class DataClass:
    pass


def get_data_from_file(filename):
    file = open(filename, 'r')
    file_vars = split_vars(file.read())
    data_object = DataClass()
    for var_line in file_vars:
        add_var(var_line, data_object)
    return data_object


def split_vars(string_from_file):
    lines = string_from_file.splitlines()
    curr = []
    file_vars = []
    for line in lines:
        if line == '':
            file_vars.append(curr)
            curr = []
        else:
            curr.append(line)
    file_vars.append(curr)
    return file_vars


def add_var(var_lines, data_object):
    var = []
    for line in var_lines[1:]:
        var.append(line_to_list(line))
    setattr(data_object, var_lines[0], var)


def line_to_list(line):
    l = []
    numbers = line.split(' ')
    for number in numbers:
        l.append(float(number))
    return l


def get_data_files(data_path):
    from os import listdir
    from os.path import isfile, join
    ps = data_path+"/"
    return [ps+f for f in listdir(data_path) if isfile(join(data_path, f))]