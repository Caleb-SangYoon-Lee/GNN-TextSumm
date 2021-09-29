# -*- coding: utf-8 -*-

import traceback

def whoami():
    stack = traceback.extract_stack()
    file_name, codeline, func_name, text = stack[-2]
    return func_name

def whereami(depth=0):
    stack = traceback.extract_stack()
    file_name, code_line, func_name, text = stack[-2 + depth]
    return '@%s[%dth line in %s]' % (func_name, code_line, file_name)

def main():
    where = whereami()
    print(f'target location in source:{where}')
    pass


if __name__ == '__main__':
    main()
    pass
