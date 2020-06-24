import time
import datetime
import random
import os
# import profile
# import cProfile, pstats, StringIO
import re
import string
import argparse

# from multiprocessing import Process, Queue, current_process, freeze_support

import multiprocessing as mp
import numpy as np

from timeit import default_timer as timer
from copy import copy
from numba import vectorize
from itertools import islice, groupby
from sklearn.preprocessing import normalize
from collections import deque as dq

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# import tensorflow as tf

''' version incorporating trained neural net prediction'''


##############################
#####       Globals      #####
##############################

chrlen = {0: 248956422, 1: 242193529, 2: 198295559, 3: 190214555, 4: 181538259, 5: 170805979, 6: 159345973, 7: 145138636, 8: 138394717, 9: 133797422, 10: 135086622, 11: 133275309, 12: 114364328, 13: 107043718, 14: 101991189, 15: 90338345, 16: 83257441, 17: 80373285, 18: 58617616, 19: 64444167, 20: 46709983, 21: 50818468, 22: 156040895, 23: 57227415, 24: 16569} #hg38_chr.length values
listList = [[] for k in chrlen.keys()]

alignLen = 36 # update this to interpret the cigar string or alignment score to give more accurate coverage
# window = int(alignLen*4.5)
fragLen = 176
window = fragLen


totalSumCoverage = mp.Value('L', 0)
rpmFactor = mp.Value('f', 0.0)

##############################
#####     Functions      #####
##############################

class Truth:
    def __init__(self, truthFile):
        self.truthFile = truthFile
        self.truthDict = {}

    def readTruth(self):
        print_info('Reading truth file', logFile)
        with open(self.truthFile, 'r') as f:
            lines = f.readlines()
            l = 0
            for line in lines:
                if not line.startswith('@'):
                    l += 1
                    line = line.strip().split('\t')
                    thisID = line[0]
                    thisChr = line[2]
                    thisPos = line[3]
                    self.storeTruth(thisID, thisChr, thisPos)
                    if l % 1000000 == 0:
                        print_info('Read and stored ' + str(l) + ' lines of truth file', logFile)
        del lines

    def storeTruth(self, ID, chrom, pos):
        ID = ID.split(':')[2:]
        try:
            self.truthDict[ID[0]][ID[1]][ID[2]] = [chrom, pos]
        except KeyError:
            try:
                self.truthDict[ID[0]][ID[1]] = {}
                self.truthDict[ID[0]][ID[1]][ID[2]] = [chrom, pos]
            except KeyError:
                # try:
                self.truthDict[ID[0]] = {}
                self.truthDict[ID[0]][ID[1]] = {}
                self.truthDict[ID[0]][ID[1]][ID[2]] = [chrom, pos]

    def getTrueLoc(self, ID):
        ID = ID.split(':')[2:]
        try:
            truthList = self.truthDict[ID[0]][ID[1]][ID[2]]
        except KeyError:
            truthList = False
        return truthList

def now():
    return str(datetime.datetime.now())[:-7]

def print_info(string, logFile):
    # print ('\t'.join(['INFO:', now(), string]))
    # with open('/afm01/scratch/scmb/uqkupton/keras_test.log', 'a') as o:
    with open(logFile, 'a') as o:
        info = '\t'.join(['INFO:', now(), string]) + '\n'
        o.write(info)

def revcomp(dna):
    complements = str.maketrans('acgtrymkbdhvACGTRYMKBDHV-', 'tgcayrkmvhdbTGCAYRKMVHDB-')
    rcseq = dna.translate(complements)[::-1]
    return rcseq
