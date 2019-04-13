import os
import sys

def print_usage(args):
    filename = os.path.basename(sys.argv[0])
    args = ' '.join(f'[{key}]' if len(options) > 1 else key
                    for key, options in args)

    print(f"usage: {filename} {args}")

def get_tuple(x):
    return x if type(x) is tuple else (x,)

def get_arguments(**kwargs):
    args = [(key, get_tuple(options)) for key, options in kwargs.items()]

    for i, (key, options) in enumerate(args):
        if len(sys.argv) > i + 1:
            yield options[0](sys.argv[i + 1])
        elif len(options) > 1:
            yield options[1]
        else:
            print_usage(args)
            sys.exit()
