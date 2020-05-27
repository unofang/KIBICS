import sys

def releaseList(array):
   del array[:]
   del array

def clearAllValuesExceptSelection(keepValues):
    this = sys.modules[__name__]

    # keep the built-in values
    for n in dir():
        if n[0]!='_'||n not in keepValues:
            delattr(this, n)
