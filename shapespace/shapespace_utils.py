# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import numpy as np
import torch
import trimesh

def logging_print(filename, oldprint):
    '''When printing, log to file `filename` as well. Returns the wrapper wrap, which when given
    a function, will return the wrapped function'''
    def wrap(func):
        '''The wrapper function that takes the initial function as input and returns a wrapped version of it.'''
        def wrapped_print(*args,**kwargs):
            oldprint(*args, **kwargs) # Print as per usual
            with open(filename,'a') as outputfile:
                kwargs['file'] = outputfile
                oldprint(*args, **kwargs) # print to file
        return wrapped_print
    return wrap

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)