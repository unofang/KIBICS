import os, psutil, numpy as np

def memoryCheck():
    process = psutil.Process(os.getpid())
    
    print ("Currently memory usage: {} Mb".format(process.memory_info().rss/1024))
