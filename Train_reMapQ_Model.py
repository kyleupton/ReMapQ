import os, time, datetime
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

import tensorflow as tf
# from keras.layers import Input, Dense, Dropout
from math import log
from keras import layers, metrics
from keras.constraints import max_norm
from keras.models import Model
from keras import regularizers, optimizers, initializers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.utils import plot_model, multi_gpu_model

from collections import deque as dq

import multiprocessing as mp

from random import shuffle

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def now():
    return str(datetime.datetime.now())[:-7]

def print_info(string):
    print('\t'.join(['INFO:', now(), string]))

def task_waiter(taskCount, waitTime, counterQueue, taskTimer):
    items_received = 0
    last_received = -1
    while items_received < taskCount:
        if counterQueue.empty():
            if last_received < items_received:
                if taskTimer:
                    pass
                    # stop=timer()
                    # print_info('Checking for completed tasks every ' + str(waitTime) +  ' seconds: ' + str(items_received) + ' of ' + str(taskCount) + ' (' + str((items_received*100)/taskCount) + '%) have completed in ' +  str(int(stop - taskTimer)) + ' seconds')
                else:
                    print_info('Checking for completed tasks every ' + str(waitTime) +  ' seconds: ' + str(items_received) + ' of ' + str(taskCount) + ' have completed already')
                    # pass
                last_received = copy(items_received)
                # time.sleep(waitTime)
        else:
            finished = counterQueue.get()
            items_received += 1
    # print_info('All ' + str(items_received) + ' current tasks are completed')
    return items_received

def unzip(task_queue, taskCount, finished, procs, holder):
    for path in iter(task_queue.get, 'STOP'):
        while int(taskCount.value) > procs:
            pass
        with taskCount.get_lock():
            taskCount.value = taskCount.value + 1
        print('unzipping ' + path[0])
        cmd = 'gunzip -c ' + path[0] + ' > ' + path[1]
        print(cmd)
        os.system(cmd)
        finished.put('1')
        time.sleep(2)

    finished.put('STOP')

def delete_unzipped(del_queue, taskCount, holder):
    for path in iter(del_queue.get, 'STOP'):
        print('deleting ' + path)
        cmd = 'rm ' + path
        # print cmd
        os.system(cmd)
        with taskCount.get_lock():
            taskCount.value = taskCount.value - 1

def draw_plots(ID, tempVarDir, covers):

    X_plot = np.linspace(0, len(covers[0])-1, len(covers[0]))
    # bins   = np.linspace(0, len(normCovers[0])-1, len(normCovers[0]))
    # print X_plot
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    maxCover = max([max(cover) for cover in covers])*1.1

    data = covers[0]
    # print data.shape
    ax[0].fill_between(X_plot, 0, data, color='#AAAAFF')
    ax[0].text(5, maxCover * 0.9, "Wrong Cover")
    ax[0].set_ylim(0, maxCover)

    data = covers[1]
    # print data.shape
    ax[1].fill_between(X_plot, 0, data, color='#AAAAFF')
    ax[1].text(5, maxCover * 0.9, "Right Cover")
    ax[1].set_ylim(0, maxCover)

    for axi in ax.ravel():
        # axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
        # axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
        axi.set_xlim(0, len(covers[0]))
        # axi.set_ylim(-0.001, 0.01)

    # for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')

    # for axi in ax[1, :]:
    axi.set_xlabel('position')

    fig.set_size_inches(20, 20)
    plotFile = os.path.join(tempVarDir, 'plot_' + ID + '.pdf')
    fig.savefig(plotFile, bbox_inches='tight')

def setup_model():
    print('Setting up model')

    validData   = []
    validLabels = []

    cover_in = layers.Input(shape=(None, 528), dtype='float32', name='cover_in')
    cover_biDir = layers.Bidirectional(layers.LSTM(1024,  activation='relu', recurrent_activation='hard_sigmoid',
        use_bias=True, kernel_regularizer=regularizers.l2(0.0001), recurrent_regularizer=regularizers.l2(0.0001),
        dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat', name='cover_biDir')(cover_in)
    cover_biDir_out = layers.Dense(1, activation='sigmoid', name='cover_biDir_out')(cover_biDir)
    dense = layers.Dense(512)
    cover_d1 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(cover_biDir))
    dense = layers.Dense(512)
    cover_d2 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)), name='cover_d2')(dense(cover_d1))
    cover_d2_out = layers.Dense(1, activation='sigmoid', name='d2_out')(cover_d2)
    dense = layers.Dense(512)
    cover_d3 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(cover_d2))
    dense = layers.Dense(512)
    cover_d4 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(cover_d3))
    cover_output = layers.Dense(1, activation='sigmoid', name='cover_output')(cover_d4)

    movav_in = layers.Input(shape=(None, 528), dtype='float32', name='movav_in')
    movav_biDir = layers.Bidirectional(layers.LSTM(1024,  activation='relu', recurrent_activation='hard_sigmoid',
        use_bias=True, kernel_regularizer=regularizers.l2(0.0001), recurrent_regularizer=regularizers.l2(0.0001),
        dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat', name='movav_biDir')(movav_in)
    movav_biDir_out = layers.Dense(1, activation='sigmoid', name='movav_biDir_out')(movav_biDir)
    dense = layers.Dense(512)
    movav_d1 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(movav_biDir))
    dense = layers.Dense(512)
    movav_d2 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)), name='movav_d2')(dense(movav_d1))
    movav_d2_out = layers.Dense(1, activation='sigmoid', name='d2_out')(movav_d2)
    dense = layers.Dense(512)
    movav_d3 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(movav_d2))
    dense = layers.Dense(512)
    movav_d4 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(movav_d3))
    movav_output = layers.Dense(1, activation='sigmoid', name='movav_output')(movav_d4)

    delta_in = layers.Input(shape=(None, 528), dtype='float32', name='delta_in')
    delta_biDir = layers.Bidirectional(layers.LSTM(1024,  activation='relu', recurrent_activation='hard_sigmoid',
        use_bias=True, kernel_regularizer=regularizers.l2(0.0001), recurrent_regularizer=regularizers.l2(0.0001),
        dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat', name='delta_biDir')(delta_in)
    delta_biDir_out = layers.Dense(1, activation='sigmoid', name='delta_biDir_out')(delta_biDir)
    dense = layers.Dense(512)
    delta_d1 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(delta_biDir))
    dense = layers.Dense(512)
    delta_d2 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)), name='delta_d2')(dense(delta_d1))
    delta_d2_out = layers.Dense(1, activation='sigmoid', name='d2_out')(delta_d2)
    dense = layers.Dense(512)
    delta_d3 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(delta_d2))
    dense = layers.Dense(512)
    delta_d4 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(delta_d3))
    delta_output = layers.Dense(1, activation='sigmoid', name='delta_output')(delta_d4)

    main_concat = layers.concatenate([cover_d4, movav_d4, delta_d4], axis=2)
    dense = layers.Dense(1024)
    main_d1 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(main_concat))
    dense = layers.Dense(1024)
    main_d2 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)), name='main_d2')(dense(main_d1))
    main_d2_out = layers.Dense(1, activation='sigmoid', name='d2_out')(main_d2)
    dense = layers.Dense(512)
    main_d3 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(main_d2))
    dense = layers.Dense(512)
    main_d4 = layers.TimeDistributed(layers.PReLU(alpha_initializer='glorot_normal',
        alpha_regularizer=regularizers.l2(0.0001)))(dense(main_d3))
    main_output = layers.Dense(1, activation='sigmoid', name='main_output')(main_d4)

    print('Defining model')
    model = Model(inputs=[cover_in, movav_in, delta_in], outputs=[main_output, cover_output, movav_output, delta_output, cover_biDir_out, movav_biDir_out, delta_biDir_out])

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1E-8, decay=0.0001, amsgrad=False)
    adadelta = optimizers.Adadelta()
    print('Compiling model')
    model.compile(optimizer=adadelta,
                  loss='binary_crossentropy',
                  # loss='mean_squared_error',
                  loss_weights=[1., 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
                  # loss='categorical_crossentropy',

                  # metrics=['accuracy', metrics.categorical_accuracy])
                  metrics=['accuracy'])
                  # metrics=['loss'])

    return model

def tokenize(file, multi, refIndex, maxRep, div, rem):
    print(file)
    rep = 1
    # div = 5
    vectors = dq()
    expects = dq()
    with open(file, 'r') as f:
        line = f.readline()
        # while (rep<=200):
        while (line and (rep<=(maxRep*div))):      #Limit number of input lines to save some processing time. Seems to be getting enough data for training
        # while line:
            ve = []
            ex = []
            # if (rep%div==12):
            if (rep%div==rem):
                for m in range(int(multi)):
                    line = line.strip()
                    lists = line.split('|')
                    # print 'len = ' + str(len(lists))
                    # print lists[0]
                    # print lists[2]
                    pos = lists[refIndex].split(',')
                    # print len(pos)
                    pos = [log(1+float(x)) for x in pos]
                    # print pos
                    ve.extend(pos)
                    ex.extend(lists[0])
                    line = f.readline()
                vectors.append(ve)
                expects.append(ex)
                line = f.readline()

            else:
                for m in range(int(multi)):
                    line = f.readline()
                line = f.readline()

            rep+=1


        # line = f.readline()
        # for line in lines:
        #     line = line.strip().split('\t')
        #     # print line
        #     line = [log(1+float(x)) for x in line]
        #     vectors.append(line)
        #     expects.append(expect)
        # # positions = [float(i) for i in range(len(vectors[0]))]
    # print expects
    # print len(vectors)
    # print len(expects)
    return vectors, expects


trainDataDir = '/nvmescratch/uqkupton/temp_'
# trainDataDir = '/bak/TN-MBDSeq/remap_test/temp_'

saveDir =  '/afm01/scratch/scmb/uqkupton/NN_logs/'
# saveDir =  '/bak/TN-MBDSeq/remap_test/NN_logs/'

numbers = ['1090284614', '1273424096', '140413783', '2464181600', '2674258102']   # These are a bunch of random numbers chosen for the initial data generation. They just work as IDs

model = setup_model()
# modelFile = '/afm01/scratch/scmb/uqkupton/models/lstm_300_adadelta_val_acc_02-01-1.00.hdf5'
# model = load_model(modelFile)


maxReps = [1000, 2000, 5000, 10000]
divs    = [5,     4,    3,    1   ]
rems    = [1,     0,    1,    0   ]

maxReps = [1000,10000]
divs    = [5,   1   ]
rems    = [1,   0   ]
multis  = [200, 500]
epochLst= [30,  100]
patLR   = [3,   5]
patES   = [7,   12]

procs = 10
task_queue = mp.Queue()
taskCount  = mp.Value('i',0)
finished   = mp.Queue()
del_queue  = mp.Queue()

for proc in range(procs):
        mp.Process(target=unzip, args=(task_queue, taskCount, finished, procs, False)).start()
        print('started unzip process ' + str(proc))
        # time.sleep(0.1)

for proc in range(procs):
        mp.Process(target=delete_unzipped, args=(del_queue, taskCount, False)).start()
        print('started delete process ' + str(proc))

scratch = '/nvmescratch/uqkupton/'
# scratch = '/ext/TN-MBDSeq/remap_test/'

for h in range(len(maxReps)):

    theseMultis = [i for i in range(2,multis[h])]
    shuffle(theseMultis)
    shuffle(theseMultis)

    for i in theseMultis:
        for number in numbers:
            thisPath = scratch +  'temp_' + number + '/train' + str(i) + '.txt.gz'
            thatPath = scratch +  'temp_' + number + '/train' + str(i) + '.txt'
            # print('thisPath = ' + thisPath)
            task_queue.put([thisPath, thatPath])
    time.sleep(60)

    for i in theseMultis:    # Set up trainng iterator to run over training data from 2 to 25 possible decisions
        multi = str(i)       # String representation of number of inputs to decide between

        maxRep = maxReps[h]
        div = divs[h]
        rem = rems[h]

        cover  = dq()
        movav  = dq()
        delta  = dq()
        labels = dq()

        allTrain  = []

        allCover  = []
        allMA     = []
        allDelta  = []
        allLabels = []
        lens = []

        for number in numbers:      #Read in the training data through the vectorise function from a file containing the labelled training data. Labels go to the label variable
            trainFile = trainDataDir + number + '/train' + multi + '.txt'

            if os.path.isfile(trainFile):

                coverTrain, thisLabels = tokenize(trainFile, multi, 2, maxRep, div, rem)
                allCover.append(coverTrain)

                maTrain, thisLabels    = tokenize(trainFile, multi, 4, maxRep, div, rem)
                allMA.append(maTrain)

                deltaTrain, thisLabels = tokenize(trainFile, multi, 6, maxRep, div, rem)
                allDelta.append(deltaTrain)

                allLabels.append(thisLabels)
                lens.append(len(thisLabels))
                print(len(thisLabels))

        minLen = min(lens)

        print('Collating and splicing data')

        for x in range(minLen):                             # Mix together the training data. Not sure if this helps or is needed, but should even out training if there is unexpected bias in one of the training sets.
            for y in range(len(allCover)):
                cover.append(allCover[y].popleft())
                movav.append(allMA[y].popleft())
                delta.append(allDelta[y].popleft())
                labels.append(allLabels[y].popleft())

        print('Reshaping cover')
        cover = np.array(cover, dtype=np.float32).reshape(minLen*len(lens), int(multi), 528)
        print('Reshaping movav')
        movav = np.array(movav, dtype=np.float32).reshape(minLen*len(lens), int(multi), 528)
        print('Reshaping delta')
        delta = np.array(delta, dtype=np.float32).reshape(minLen*len(lens), int(multi), 528)
        # labels = np.array(labels, dtype=np.float32).reshape(minLen*len(lens), int(multi), 1)
        # labels = np.array(labels, dtype=np.int32).reshape(minLen*len(lens), int(multi))
        print('Reshaping labels')
        labels = np.array(labels, dtype=np.int32).reshape(minLen*len(lens), int(multi), 1)


        filepath    = '/afm01/scratch/scmb/uqkupton/NN_logs/lstm' + '_' + str(i) + "_adadelta_val_acc_0 2-{epoch:02d}-{val_main_output_acc:.2f}.hdf5"
        # filepath    = '/bak/TN-MBDSeq/remap_test/NN_logs/lstm' + '_' + str(i) + "_adadelta_val_acc-{epoch:02d}-{val_main_output_acc:.2f}.hdf5"
        checkpoint  = ModelCheckpoint(filepath, monitor='val_main_output_acc', verbose=1, save_best_only=True, save_weights_only=False, period=1)
        earlyStop   = EarlyStopping(monitor='val_main_output_acc', min_delta=0.0001, patience=12, verbose=1, mode='auto')
        reduceLR    = ReduceLROnPlateau(monitor='val_main_output_acc', factor=0.2, patience=5, verbose=1, mode='auto', min_lr=0.000000001)
        # tensorBoard = TensorBoard(log_dir='/bak/NN_logs', histogram_freq=2, batch_size=16, write_graph=False)#, write_grads=False, write_images=False, embeddings_freq=2)#, embeddings_layer_names=None, embeddings_metadata=None)

        print('Fitting model')
        model.fit([cover, movav, delta], [labels, labels, labels, labels, labels, labels, labels], epochs=epochLst[h], batch_size=32, validation_split=0.5, callbacks=[earlyStop, reduceLR, checkpoint])#, validation_data=(validData, validLabels))  # starts training


        predictions = model.predict([cover, movav, delta])


        # Run a basic assessment of the performance with the input data. The performance on a completely separate data set is done in a different script (keras_predictor_03.py) as it takes quite a while to read in enough data.
        rightY = dq()
        rightN = dq()
        wrongY = dq()
        wrongN = dq()
        passY  = dq()
        passN  = dq()

        rightD = dq()
        wrongD = dq()
        correct = 0
        incorrect = 0
        passed = 0
        ambig = 0

        for p in range(len(predictions[0])):
            ans = np.argmax(labels[p])
            pred = np.argmax(predictions[0][p])
            predValue = predictions[0][p][pred]
            remain = np.delete(predictions[0][p], pred)
            # if ((predictions[0][p][pred] - max(remain)) >= 0.65 ):
            if ((predValue - max(remain)) > 0.0 ):
                if (ans == pred):# and not (max(predictions[0][p]) == max(remain)):
                    rightY.append(predValue)
                    for v in remain:
                        correct += 1
                        rightD.append(predValue - v)
                        rightN.append(v)
                elif (predValue == max(remain)):
                    ambig += i-1
                else:
                    remainLess = np.array([r for r in remain if r < predictions[0][p][ans]]) # sub decisions that were still correct calls
                    remainMore = np.array([r for r in remain if r >= predictions[0][p][ans]])# sub decisions that were incorrect
                    wrongY.append(predValue)
                    for l in remainLess:
                        correct += 1
                        rightD.append(predValue - l)
                        rightN.append(l)
                    for m in remainMore:
                        incorrect += 1
                        wrongD.append(m - predictions[0][p][ans])
                        wrongN.append(m)

            else:
                passed += i-1
                # if (passed < 50):
                #     draw_plots(multi + '_' + str(p), 'pdf', cover[p])

        print('correctly predicted ' + str(correct) + ' of ' + str(correct + incorrect) + ' ' + str((correct*100.0)/(correct + incorrect)) + '%')
        print('ambiguous = ' + str(ambig) + ', passed = ' + str(passed))



        for number in numbers:
            thisPath = os.path.join(scratch, 'temp_' + number, 'train' + str(i) + '.txt')
            del_queue.put(thisPath)


for proc in range(procs):
    task_queue.put('STOP')
    del_queue.put('STOP')