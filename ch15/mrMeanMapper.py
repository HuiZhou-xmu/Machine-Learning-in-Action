# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:10:11 2017

@author: ZH
"""

import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()

input_data = read_input('inputFile.txt')
#input = read_input(sys.stdin)
input_data = [float(line) for line in input_data]
numInputs = len(input_data)
input_data = mat(input_data)
sqInput = power(input,2)
print('%d\t%f\t%f' % (numInputs, mean(input_data), mean(sqInput)))
#print(>> sys.stderr, 'report: still slive')