
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

def cigar_to_list(cigar):
    cigList = []
    cig_iter = groupby(cigar, lambda c: c.isdigit())
    for g,n in cig_iter:
        cigList.append([int("".join(n)), ",".join(next(cig_iter)[1])])
    # print cigList
    lastPos = 0
    sliceList = [[],[]]
    for c in cigList:
        # print c
        thisPos = lastPos + int(c[0])
        if c[1].upper() == 'H':
            sliceList[0].append([lastPos,thisPos])
        elif c[1].upper() == 'D':
            thisPos -= c[0]
            # pass
        else:
            sliceList[1].append([lastPos,thisPos])
            pass
        lastPos = thisPos
    # print sliceList
    sliceList[0].reverse()
    length = 0
    for l in sliceList[1]:
        thislen = l[1]-l[0]
        length += thislen
    sliceList[1] = length
    return sliceList

def make_empty_arrays():
    start = timer()
    global listList
    for k,v in chrlen.items():
        listList[k] = mp.RawArray('i', v)
    end = timer()

def run_temp_files(window, alignLen, tempdir, levels, truth, fragLevels, predQueue, predDict, logFile):
    tempFiles = [x for x in os.listdir(tempdir) if (x.endswith('.tmpdedup'))]
    tempFiles = sorted(tempFiles, key=lambda x: int(x.split('.')[0]))
    for file in tempFiles:
        start = timer()
        multiple = int(file.split('.')[0])

        outTrain = os.path.join(tempdir,'train' + str(multiple) + '.txt')
        outTrue  = os.path.join(tempdir,'truth' + str(multiple) + '.txt')

        print_info('Now processing file ' + os.path.join(tempdir, file), logFile)

        #build initial coverage using unique mapping reads
        if file == '1.tmpdedup':
            n = 20000000
            with open(os.path.join(tempdir,file), 'r') as f:
                while True:
                    next_n_lines = list(islice(f, n))
                    if not next_n_lines:
                        break
                    print_info('File ' + file + ' contains ' + str(len(next_n_lines)) + ' reads', logFile)
                    process_singles(next_n_lines, logFile)
            end = timer()
            print_info('Processing file  ' + file + ' took '+ str(end - start) + ' seconds', logFile)

        # decide on best location for multimapping reads, processing files in order of the number of best scoring matches found.
        elif multiple <= levels:
            n = (int(1500000/multiple)) * multiple
            outSam = os.path.join(tempdir, str(multiple) + '_remap.sam')

            if os.path.isfile(outSam):
                os.system('rm ' + outSam)

            thisSlice = 0
            with open(os.path.join(tempdir,file), 'r') as f:
                while True:
                    start1 = timer()
                    next_n_lines = list(islice(f, n))
                    if not next_n_lines:
                        break
                    if len(next_n_lines) == n:
                        print_info('Slice ' + str(thisSlice + 1) + ' of file ' + file + ' contains ' + str((len(next_n_lines)*1.0)/multiple) + ' sets of ' + str(multiple) + ' reads', logFile)
                    else:
                        # print_info('Slice ' + str(thisSlice + 1) + ' of file ' + file + ' contains ' + str((len(next_n_lines)*1.0)/multiple) + ' sets of ' + str(multiple) + ' reads', logFile)
                        print_info('File ' + file + ' contains ' + str(((n*thisSlice)+len(next_n_lines))/(multiple*1.0)) + ' sets of ' + str(multiple) + ' reads', logFile)
                    process_multis(next_n_lines, multiple, window, alignLen, outSam, tempdir, truth, fragLevels, levels, outTrain, outTrue, predQueue, predDict, logFile)
                    end = timer()
                    if len(next_n_lines) == n:
                        print_info('Processing file  ' + file + ' slice ' + str(thisSlice + 1) + ' took '+ str(end - start1) + ' seconds', logFile)

                    print_info('Processing file  ' + file + ' took '+ str(end - start) + ' seconds', logFile)
                    thisSlice += 1

def process_singles(lines, logFile):
    start3 = timer()
    global totalSumCoverage
    global rpmFactor
    global listList

    #Set function variables
    allTasks = len(lines)
    procs = mp.cpu_count()
    procs = 24

    # Create queues
    task_queue          = mp.Queue()                                #Queue for initial sublists of input bed lines
    chrom_queues        =[mp.Queue() for x in range(len(chrlen))]  #List of queues, with a queue for each chromosome. Each queue will have a list of lists, with up to one list added from each sublist.
    combinedCoverQueue  = mp.Queue()                                #Queue for lists of chromosome and complete coverage list pair. There should be one entry per chromosome
    finished_align      = mp.Queue()                                #Queue for count of finished align_to_cover tasks
    chromSubCover       = mp.Queue()                                #Queue for count of coverage sublists added for individual chromosomes (Will be larger than finished align as there will be upto 1 list per chromosome per align_to_cover task).
    finished_cover      = mp.Queue()                                #Queue for counts of finished chromosome combined coverage calculations
    finished_chrom      = mp.Queue()

    # Split the list of input bed lines into smaller sublists and count the number created
    sublists = 0
    step = 250000
    for c in range(int(allTasks/step)+1):
        task_queue.put(list(lines[step*c : (step*c)+step]))
        sublists += 1
    del lines
    print_info('Number of sublists = ' + str(sublists), logFile)

    #Submit the sublists for processing
    for i in range(procs):
        mp.Process(target=align_to_cover, args=(task_queue, chrom_queues, finished_align, len(chrlen), chromSubCover, 0)).start()

    #Wait for the sublists to finish processing and post updates as tasks are completed
    waitTime = 2
    task_waiter(sublists, waitTime, finished_align, start3)
    # Kill processes once all taks have completed
    for i in range(procs):
        task_queue.put('STOP')

    start2 = timer() #Timer for summing chromosome coverage
    # Start a process for each chromosome to combine the covarage outputs from align to cover into a single list per chromosome
    for i in range(len(chrlen)):
        mp.Process(target=add_chrom_cover, args=(i, chrom_queues[i], listList[i], combinedCoverQueue, finished_cover, finished_chrom, totalSumCoverage, rpmFactor, True)).start()

    #Get the total number of lists created by align_to_cover (up to one per chromosome per sublist)
    chromSubCounter = 0
    while not chromSubCover.empty():
        chromSubCover.get()
        chromSubCounter += 1

    waitTime = 10
    task_waiter(chromSubCounter, waitTime, finished_cover, start2)

    # Kill processes for add_chrom_cover
    for i in range(len(chrom_queues)):
        chrom_queues[i].put('STOP')

    totalMapped = 0
    totalSumCover = 0

    for i in range(len(chrlen)):
        mapped = np.count_nonzero(listList[i])
        totalMapped += mapped
        sumCoverage = np.sum(listList[i])
        totalSumCover += sumCoverage

    rpmFactor1 = 1.0/(float(totalSumCoverage.value)/1000000.0)

def process_multis(lines, multiple, window, alignLen, outSam, tempdir, truth, fragLevels, levels, outTrain, outTrue, predQueue, predDict, logFile):
    global totalSumCoverage
    global rpmFactor
    global listList

    #Set function variables
    lock = mp.Lock()
    predLock = mp.Lock()
    allTasks = len(lines)
    # procs = mp.cpu_count()
    procs = 24
    # procs = 12

    tempVarDir = tempdir + '/frags/' + str(multiple)

    if not os.path.exists(tempVarDir):
        os.makedirs(tempVarDir)

    # Create queues
    task_queue          = mp.Queue()
    chrom_queues        =[mp.Queue() for x in range(len(chrlen))]   #List of queues, with a queue for each chromosome. Each queue will have a list of lists, with up to one list added from each sublist.
    combinedCoverQueue  = mp.Queue()                                #Queue for lists of chromosome and complete coverage list pair. There should be one entry per chromosome
    finished_var        = mp.Queue()                                #Queue for count of finished calc_rel_var tasks
    chrom_sub_var       = mp.Queue()                                #Queue for count of coverage sublists added for individual chromosomes (Will be larger than finished align as there will be upto 1 list per chromosome per align_to_cover task).
    finished_cover      = mp.Queue()                                #Queue for counts of finished chromosome combined coverage calculations
    finished_chrom      = mp.Queue()
    written             = mp.Queue()

    varTaskCount  = 0

    # targ = 30000
    targ = 8000
    # targ = 800

    taskTarget = int(len(lines)/targ)
    if (len(lines) > targ):
        if (len(lines)%targ != 0):
            taskTarget += 1

    if taskTarget == 0:
        taskTarget += 1

    if taskTarget < procs:
        pass
    elif (procs < taskTarget < (2*procs)):
        taskTarget = procs*2

    stepLen = int((len(lines)/multiple)/taskTarget)*multiple
    if (len(lines)%stepLen != 0):
        stepLen += multiple

    for b in range(taskTarget):
        task_queue.put(list(lines[b*stepLen : (b*stepLen)+stepLen]))
        varTaskCount += 1
    del lines

    print_info('Process Multi step size = ' + str(stepLen) + ' (' + str(varTaskCount) + ' subtasks of ' + str((stepLen*1.0)/multiple) + ' multis)', logFile)

    start2 = timer()
    for i in range(procs):
        subTruth = truth
        mp.Process(target=calc_rel_var, args=(multiple, window, fragLen, task_queue, chrom_queues, chrom_sub_var, finished_var, lock, predLock, i, outSam, written, tempdir, subTruth, fragLevels, levels, outTrain, outTrue, predQueue, predDict)).start()

    # Start a process for each chromosome to combine the covarage outputs from align to cover into a single list per chromosome
    for i in range(len(chrlen)):
        mp.Process(target=add_chrom_cover, args=(i, chrom_queues[i], listList[i], combinedCoverQueue, finished_cover, finished_chrom, totalSumCoverage, rpmFactor, False)).start()

    waitTime = 2
    task_waiter(varTaskCount, waitTime, finished_var, start2)

    start2 = timer() #Timer for summing and returning chromosome coverage

    for i in range(procs):
        task_queue.put('STOP')
    # print_info('Put stops into task queue')

    #Get the total number of lists created by calc_rel_var (up to one per chromosome per sublist)
    chromSubVarCounter = 0
    while not chrom_sub_var.empty():
        chrom_sub_var.get()
        chromSubVarCounter += 1

    # Wait for all coverage summation calculations (add_chrom_cover) to complete
    # waitTime = 10
    task_waiter(chromSubVarCounter, waitTime, finished_cover, start2)

    # Kill processes for add_chrom_cover
    for i in range(len(chrom_queues)):
        chrom_queues[i].put('STOP')
    # print_info('Put stops into done queue')

    returned = 0
    while returned < len(chrlen): # not combinedCoverQueue.empty():
        result = finished_chrom.get()
        returned += 1

    written_count = 0
    while written_count < varTaskCount: #not write_queue.empty():
        result = written.get()
        written_count += 1

def get_start_pos(flag, posn, fragLen):
    binFlag = bin(int(flag))
    if len(flag) >= 7:
        revFlag = int(binFlag[-5])
    else:
        revFlag = 0

    if revFlag:
        adjustedPos = posn + alignLen - fragLen
        return adjustedPos
    else:
        return posn

def calc_rel_var(multiple, window, fragLen, input, output, chrom_sub_var, finished_var, lock, predLock, proc, outSam, written, tempdir, truth, fragLevels, levels, outTrain, outTrue, predQueue, predDict):
    global rpmFactor
    global listList

    # pr = cProfile.Profile()
    # pr.enable()

    # counter   = 1
    coverList = get_cover_list(False, window, fragLen)
    for lines in iter(input.get, 'STOP'):
        writeFieldsLists = []
        theseTasksCount = len(lines)
        results = [[] for c in range(len(chrlen))]

        trainData  = []
        trainTruth = []

        predData = dq()

        compWindow = (window*2) + fragLen
        for i in range(int(theseTasksCount/multiple)):
            multiLines  = lines[i*multiple : (i*multiple)+multiple]
            multifields = []
            # negCovers   = np.zeros(shape=(multiple,compWindow), dtype=np.int32)
            posCovers   = np.zeros(shape=(multiple,compWindow), dtype=np.int32)
            multiIter = 0
            for line in multiLines:
                fields   = line.strip().split()
                multifields.append(fields)
                ID       = fields[0]
                flag     = fields[1]
                chrom    = get_chrom(fields[2])
                posn     = int(fields[3])-1
                pos      = get_start_pos(flag, posn, fragLen)
                negCover = np.array(listList[chrom][ pos-window : pos+fragLen+window], dtype=np.int32)
                if len(negCover) < compWindow:
                    negCover = np.lib.pad(negCover, (0,compWindow - len(negCover)), 'constant', constant_values=(0))
                posCover = combine_cover_lists(negCover, coverList)
                # negCovers[multiIter] = negCover
                posCovers[multiIter] = posCover
                multiIter += 1

            rpm = copy(rpmFactor.value)
            # negCovers = negCovers * rpm
            posCovers = posCovers * rpm
            # assert posCovers

            # covers = np.append(negCovers, posCovers).reshape(multiple*2, compWindow)

            # if (multiple <= int(levels)):
            #     tempVarDir  = os.path.join(tempdir, 'frags', str(multiple))
            #     tempVarFile = os.path.join(tempVarDir, (ID + '.txt'))

            #     if (multiple <= int(fragLevels)):
            #         with open(tempVarFile, 'w') as t:
            #             for f in range(len(multifields)):
            #                 t.write(multifields[f][0])
            #                 t.write(', ')
            #             t.write('\n')
            #             t.write(str(covers.tolist()))
            #             t.write('\n')
            # else:
            #     tempVarFile = False

            tempVarFile = False


            # maCovers, maDeltas = calc_ma_matrix(multiple, covers, window, compWindow, tempVarFile) # 2xmultiple Matrix of variance values
            maCovers, maDeltas = calc_ma_matrix(multiple, posCovers, window, compWindow, tempVarFile) # 2xmultiple Matrix of variance values

            predData.append([posCovers, maCovers, maDeltas])       # This currently contains both neg and pos cover data. This could prob be changed to just use pos cover for less memory usage

            # add cover data to prediction queue. Prediction function will run predictions and add results to a dictionary item that will be checked until result is present

        # if (ID in ['GAPC:2:3:7527:10491', 'GAPC:2:25:8206:13248', 'GAPC:2:28:13475:3761', 'GAPC:2:50:3180:7447']):
        #     print ID
        #     print predData
        #     print len(predData)

        # if ((theseTasksCount/multiple) != len(predData)):
        #     print predData
        #     print ID + '\t' + str(len(predData)) + '\t' + str(theseTasksCount/multiple) + '\t' + str(len(lines))

        # print_info('predData length = ' + str(len(predData)) + '\t' + str(theseTasksCount/multiple) + '\t' + str(len(lines)) + '\t' + ID)

        try:
            if ID:
                pass
        except UnboundLocalError:
            # print lines
            # print len(lines)
            pass

        if lines:
            predQueue.put([ID, multiple, predData])

            waiting = True
            while waiting:
                try:
                    predictions = predDict[ID]
                    if predictions:
                        # time.sleep(0.1)
                        # predictions = predDict[ID]
                        waiting = False
                        del predDict[ID]
                except:
                    time.sleep(0.1)
                    pass

            for i in range(int(theseTasksCount/multiple)):
                multiLines  = lines[i*multiple : (i*multiple)+multiple]

                try:
                    bestIndex = predictions[i]
                except IndexError:
                    # print 'i = ' + str(i) + '\t' + ID
                    # print 'predictions = ' + str(predictions)
                    # bestIndex = 0
                    pass

                multifields = []
                for line in multiLines:
                    fields   = line.strip().split()
                    multifields.append(fields)

                chrom     = get_chrom(multifields[bestIndex][2])
                pos       = int(multifields[bestIndex][3])-1
                smlCoverList  = coverList[window:window+fragLen]
                results[chrom].append([chrom,pos,smlCoverList])

                writeFieldsLists.extend(convert_bits(multifields,bestIndex))
                # counter += 1

            for l in range(len(results)):
                output[l].put(results[l])
                chrom_sub_var.put('1')
            # trainQueue.put([trainData,trainTruth])
            # write header to file also
            lock.acquire()
            write_SAM(writeFieldsLists, outSam, written)
            # if multiple > 25:
            #     write_training(multiple, levels, trainData, outTrain)
            lock.release()
        else:
            written.put('1')

        finished_var.put('1')


    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats(30)
    # # print 'ma'
    # print s.getvalue()



def nnPredictor(predQueue, predDict, modelFile, device, portion, nnLock):
    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot as plt


    nnLock.acquire()

    import tensorflow as tf
    from math import log
    from keras import layers, metrics
    from keras.constraints import max_norm
    from keras.models import Model, load_model
    from keras import regularizers, optimizers, initializers
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
    from keras.utils import plot_model, multi_gpu_model

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = portion
    config.gpu_options.visible_device_list = device
    set_session(tf.Session(config=config))
    with tf.device('/device:GPU:' + device):

        # modelFile = '/afm01/scratch/scmb/uqkupton/models/lstm_25_val_acc-01-0.96.hdf5'
        model = load_model(modelFile)
        model.load_weights(modelFile)
        global window
        global fragLen
        compWindow = (window*2) + fragLen

        nnLock.release()


        for inData in iter(predQueue.get, 'STOP'):
            ID = inData[0]
            multiple = inData[1]
            predData = inData[2]
            preds = [-1 for x in range(len(predData))]

            cover = np.zeros(shape=(len(predData), multiple, compWindow), dtype=np.float64)
            movav = np.zeros(shape=(len(predData), multiple, compWindow), dtype=np.float64)
            delta = np.zeros(shape=(len(predData), multiple, compWindow), dtype=np.float64)

            for i in range(len(predData)):
                cover[i] = predData[i][0]#[multiple:]
                movav[i] = predData[i][1]#[multiple:]
                delta[i] = predData[i][2]#[multiple:]

            predictions = model.predict([cover, movav, delta])
            for p in range(len(predictions[0])):
                pred = np.argmax(predictions[0][p])
                preds[p] = pred

            if (-1 in preds):
                print_info(inData, logFile)
                print_info(predictions, logFile)
                print_info(preds, logFile)
                print_info(ID, logFile)
            predDict[ID] = preds

def write_SAM(samLists, outSam, written):
    outlines = ''
    for sam in samLists:
        outlines += '\t'.join(sam) + '\n'
    with open(outSam, 'a') as o:
        o.write(outlines)
        written.put('1')

def write_training(multiple, levels, trainData, outTrain):
    global window
    global fragLen

    blank = [['0']]
    blank.extend([['0' for x in range(window+fragLen+window)] for y in range(4)])
    # print 'Blank length = ' + str(len(blank))
    trainLine = ''
    # print 'trainLine length = ' + str(len(trainLine))
    for trainSet in trainData:
        thisTrain = copy(trainSet)
        # print 'train length = ' + str(len(train))
        # for l in range(levels-len(thisTrain)):
        #     thisTrain.append(blank)
        # for m in range(multiple):
        for t in thisTrain:
            # print 't length = ' + str(len(t))
            # print t
            # print t[0]
            test = [','.join(t[u]) for u in range(len(t))]
            # print test
            trainLine += '|'.join([','.join(t[u]) for u in range(len(t))]) + '\n'
        trainLine += '--\n'
            # random.shuffle(thisTrain)
    with open(outTrain, 'a') as o:
        o.write(trainLine)

def adjust_read(read, qual, adjusted, adjuster):
    # print adjusted
    if adjusted != []:
        for adj in adjusted:
            if adj[0] == 0:
                read = 'N'*adj[1] + read
                qual = '#'*adj[1] + qual
            else:
                read = read + 'N'*(adj[1]-adj[0])
                qual = qual + '#'*(adj[1]-adj[0])
    # print adjuster
    if adjuster != []:
        for a in adjuster:
            if a[0] == 0:
                read = read[a[1]:]
                qual = qual[a[1]:]
            else:
                read = read[:a[0]]
                qual = qual[:a[0]]
    return [read, qual]

def convert_bits(multifields, bestIndex):

    bits = bin(int(multifields[0][1]))
    if len(bits) >= 7:
        origOrient = bits[-5] #set orientation for original read from orientation bit in flags field
    else:
        origOrient = '0'

    for i in range(len(multifields)):
        if i == bestIndex:
            multifields[i][1] = str(int(multifields[i][1]) & ~(1<<8)) #Set as not secondary alignment
            if len(multifields[i][1]) >= 7:
                bestOrient = bin(int(multifields[i][1]))[-5]
            else:
                bestOrient = '0'
        else:
            multifields[i][1] = str(int(multifields[i][1]) | 1<<8) #Set as Secondary Alignment

    #Re-order list to put best index first
    if not bestIndex == 0:
        read = multifields[0][9]
        qual = multifields[0][10]
        if not origOrient == bestOrient:
            read = revcomp(read)
            qual = qual[::-1]

        firstCigar = multifields[0][5]
        cigar = multifields[bestIndex][5]
        adjusted = cigar_to_list(firstCigar)
        adjuster = cigar_to_list(cigar)
        if adjusted[1] != adjuster[1]:
            read, qual = adjust_read(read, qual, adjusted[0], adjuster[0])

        multifields[bestIndex][9] = read
        multifields[bestIndex][10] = qual
        multifields[0][9]  = '*'
        multifields[0][10] = '*'
        reOrdered = [multifields[bestIndex]]
        reOrdered.extend(multifields[:bestIndex])
        reOrdered.extend(multifields[bestIndex+1:])
        return reOrdered
    else:
        return multifields
    # writeFieldsLists.extend(multifields)

def get_best_index(varMatrix, multiple):
    sumVars = []
    for v in range(multiple):
        sumVars.append(varMatrix[1][v] + np.sum(np.delete(varMatrix[0], v)))               ###XXXXXXXXXXXXXXXXXXXXX
    bestIndex = np.argmin(sumVars)
    return bestIndex

def calc_ma_matrix(multiple, covers, window, compWindow, tempVarFile):
    # maCovers = np.array([moving_average(cover, n=7) for cover in covers]).reshape(multiple, compWindow)
    maCovers = np.array([moving_average(cover, n=31) for cover in covers]).reshape(multiple, compWindow)
    # maCovers = np.array([moving_average(cover, n=61) for cover in covers]).reshape(multiple, compWindow)
    maCovers = np.array([moving_average(cover, n=121) for cover in maCovers]).reshape(multiple, compWindow)
    deltas = np.absolute(covers - maCovers)
    # assert maCovers
    # assert deltas
    return [maCovers, deltas]

def moving_average(a, n=121):
    result = []
    if n%2 == 0: ## Make sure n is not an even number, otherwise array returned will be 1 digit shorter than input
        n += 1
    ret = np.cumsum(a, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    av = ret[n - 1:] / n
    head = np.array([av[0] for i in range(int(n/2))])
    tail = np.array([av[-1] for i in range(int(n/2))])
    result.extend(head)
    result.extend(av)
    result.extend(tail)
    result = np.array(result)
    return result

# def tf_moving_average(a, n=121):
#     if n%2 == 0: ## Make sure n is not an even number, otherwise array returned will be 1 digit shorter than input
#         n += 1
#     result = []
#     t = tf.convert_to_tensor(a, dtype=tf.float32)
#     ret = tf.cumsum(t)
#     ret[n:] = ret[n:] - ret[:-n]
#     dif = tf.subtract(tf.slice(ret,n,ret.get_shape()[0]), tf.slice(ret,0,ret.get_shape()[0]-n))
#     av = tf.div(dif,n)

#     # ret = np.cumsum(a, dtype=np.float64)
#     # ret[n:] = ret[n:] - ret[:-n]
#     # av = ret[n - 1:] / n
#     head = np.array([av[0] for i in range(n/2)])
#     tail = np.array([av[-1] for i in range(n/2)])
#     result.extend(head)
#     result.extend(av)
#     result.extend(tail)
#     result = np.array(result)
#     # ha = np.concatenate(head, av)
#     return result

def align_to_cover(input, output, finished_align, chromCount, chromSubCover, thisWindow):
    global fragLen
    for lines in iter(input.get, 'STOP'):
        results = [[] for c in range(chromCount)]
        # print len(lines)
        for line in lines:
            # print line
            fields = line.strip().split()
            # print fields
            chrom    = get_chrom(fields[2])
            posn     = int(fields[3])-1
            flag     = fields[1]
            pos      = get_start_pos(flag, posn, fragLen)
            coverList = get_cover_list(False, thisWindow, fragLen)
            results[chrom].append([chrom,pos-1-thisWindow,coverList])

        for l in range(len(results)):
            output[l].put(results[l])
            chromSubCover.put('1')
        finished_align.put('1')

def add_chrom_cover(chrom, chromQueue, chromList, combinedCoverQueue, finished_cover, finished_chrom, totalSumCoverage, rpmFactor, first):
    for coverList in iter(chromQueue.get, 'STOP'):
        coverCount = 0
        for cover in coverList:
            thisChrom = cover[0]
            if not thisChrom == chrom:
                # print cover
                # print 'Wrong chromosome in list!!!'
                # time.sleep(1)
                pass
            pos = cover[1]
            coverage = cover[2]
            prevCover = chromList[pos:pos+len(coverage)]
            chromList[pos:pos+len(coverage)] = combine_cover_lists_min(prevCover, coverage)
            coverCount += 1
        finished_cover.put('1')
        with totalSumCoverage.get_lock():
            totalSumCoverage.value += coverCount
        with rpmFactor.get_lock():
            if totalSumCoverage.value == 0:
                rpmFactor.value = 0
            else:
                rpmFactor.value  = 1.0/(float(totalSumCoverage.value)/1000000.0)

    # print chrom
    # print chromList[1000000:1500000]


    finished_chrom.put(chrom)

def get_cover_list(MDtag, win, aLen):
    if MDtag:
        while '^' in MDtag:
            MDtag = MDtag.replace('^','')
        MD = re.split('[A-Z,a-z]', MDtag)
        MDlist = [int(x) if x != '' else None for x in MD]
        while None in MDlist:
            MDlist.remove(None)
        thisSum = sum(MDlist) + len(MDlist)-1
        zeros = [0]* win
        ones  = [1]* thisSum
        zeros2 = [0] * (win + (aLen - thisSum))
        coverList = np.array(zeros+ones+zeros2, dtype=np.int32)
        return coverList
    else:
        zeros = [0]* win
        ones  = [1]* aLen
        coverList = np.array(zeros+ones+zeros, dtype=np.int32)
        return coverList

def get_chrom(chrom):
    if chrom == 'chrX':
        chrom = 22
    elif chrom == 'chrY':
        chrom = 23
    elif chrom == 'chrM':
        chrom = 24
    else:
        chrom = int(chrom[3:])-1
    return(chrom)

def task_waiter(taskCount, waitTime, counterQueue, taskTimer):
    items_received = 0
    last_received = -1
    while items_received < taskCount:
        if counterQueue.empty():
            if last_received < items_received:
                if taskTimer:
                    stop=timer()
                    # print_info('Checking for completed tasks every ' + str(waitTime) +  ' seconds: ' + str(items_received) + ' of ' + str(taskCount) + ' (' + str((items_received*100)/taskCount) + '%) have completed in ' +  str(int(stop - taskTimer)) + ' seconds')
                else:
                    # print_info('Checking for completed tasks every ' + str(waitTime) +  ' seconds: ' + str(items_received) + ' of ' + str(taskCount) + ' have completed already')
                    pass
                last_received = copy(items_received)
                # time.sleep(waitTime)
        else:
            finished = counterQueue.get()
            items_received += 1
    # print_info('All ' + str(items_received) + ' current tasks are completed')
    return items_received

def combine_cover_lists(x, y):
    return x+y

def combine_cover_lists_min(x, y):
    try:
        return x + y
    except ValueError:
        if len(x) > len(y):
            x = x[:len(y)]
        if len(y) > len(x):
            y = y[:len(x)]
        return x + y

def main(args):
    inDirs = args.inDir.split(',')
    for inDir in inDirs:
        tempdir = os.path.join(args.baseDir, inDir)
        # print tempdir
        if (args.frags):
            tempFrags = os.path.join(tempdir, 'frags')
            if not os.path.isdir(tempFrags):
                cmd = 'mkdir ' + tempFrags
                # print cmd
                os.system(cmd)

        if (args.truth):
            truth = Truth(args.truth)
            truth.readTruth()
        else:
            truth = False

        main_start = timer()
        make_empty_arrays()

        GPUProcs = 6
        predQueue           = mp.Queue()
        manager             = mp.Manager()
        predDict            = manager.dict()
        allocate = [['0','0.155'], ['1','0.155'],
                    ['2','0.155'], ['3','0.155'],
                    ['4','0.155'], ['5','0.155'],
                    ['0','0.155'], ['1','0.155'],
                    ['0','0.155'], ['1','0.155'],
                    ['0','0.155'], ['1','0.155']]

        nnLock = mp.Lock()

        for i in range(GPUProcs):
            device = allocate[i][0]
            portion = 0.9#1.8/GPUProcs
            mp.Process(target=nnPredictor, args=(predQueue, predDict, args.net, device, portion, nnLock)).start()
            # time.sleep(2)

        run_temp_files(window, alignLen, tempdir, int(args.levels), truth, args.frags, predQueue, predDict, args.log)

        for i in range(GPUProcs):
            predQueue.put('STOP')

        main_end = timer()
        print_info('Processing all temp files took ' + str(main_end - main_start) + ' seconds', args.log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse RC-seq data')
    parser.add_argument('-b', '--basedir',   dest='baseDir',    default=False,  help='absolute path to base of input directory.')
    # parser.add_argument('-b', '--basedir',   dest='baseDir',    default='/data/TN-MBDSeq/remap_test',  help='absolute path to base of input directory.')
    parser.add_argument('-i', '--indir',     dest='inDir',      default=False,  help='absolute path to input directory. Multiple directories can be specified using comma as a separator.')
    # parser.add_argument('-i', '--indir',     dest='inDir',      default='temp_20',  help='absolute path to input directory. Multiple directories can be specified using comma as a separator.')
    parser.add_argument('-t', '--truth',     dest='truth',      default=False,  help='use file specified as the truth file to generate true coverage data')
    parser.add_argument('-l', '--levels',    dest='levels',     default=500,    help='Specify the maximum number of multimappings that will be considered')
    parser.add_argument('-o', '--outdir',    dest='outDir',     default=False,  help='Specify a different output directory, otherwise the input directory will be used for the final output files')
    # parser.add_argument('-o', '--outdir',    dest='outDir',     default='/data/TN-MBDSeq/remap_test/temp_20/',  help='Specify a different output directory, otherwise the input directory will be used for the final output files')
    parser.add_argument('-c', '--cleanrun',  dest='cleanRun',   default=False,  help='Remove all temporary files after run to reduce disk usage')
    # parser.add_argument('-p', '--predictor', dest='predictor',  default=False,  help='path to the predictor to use for best hit identification')
    parser.add_argument('-f', '--frags',     dest='frags',      default=False,  help='write frag files to the specified level')
    parser.add_argument('-n', '--net',       dest='net',        default='/afm01/scratch/scmb/uqkupton/models/lstm_25_val_acc-01-0.96.hdf5',  help='absolute path to trained neural net')
    # parser.add_argument('-n', '--net',       dest='net',        default='/bak/NN_logs/lstm_25_val_acc-01-0.96.hdf5',  help='absolute path to trained neural net')
    parser.add_argument('-g', '--log',       dest='log',        default=False,  help='absolute path to log file')

    args = parser.parse_args()
    main(args)
