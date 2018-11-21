
def indent(str, size=1, newline=''):
    return (' ' * (4*size)) + str + newline


def indentNL(str, size=2):
    return indent(str, size, '\n')
