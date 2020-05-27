import os, psutil, numpy as np

def memoryCheck():
    process = psutil.Process(os.getpid())
    return process.get_memory_info()[0] / float(2 ** 20)
