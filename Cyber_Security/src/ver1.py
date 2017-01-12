#! /usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import ELU
from keras.optimizers import *
from keras.utils import np_utils
from keras import backend as K

# load data

all_train = []
print("Loading train")
with open("../data/train", 'r') as f:
    for line in f:
        line = line.strip()
        all_train.append(line.split(','))

# type of attacks
dos = set(["apache2", "back", "mailbomb", "processtable", "snmpgetattack",
       "teardrop", "smurf", "land", "neptune", "pod", "udpstorm"])
u2r = set(["ps", "buffer_overflow", "perl", "rootkit", "loadmodule", "xterm",
       "sqlattack", "httptunnel"])
r2l = set(["ftp_write", "guess_passwd", "snmpguess", "imap", "spy", "warezclient",
       "warezmaster", "multihop", "phf", "named", "sendmail", "xclock", "xsnoop",
       "worm"])
probe = set(["nmap", "ipsweep", "portsweep", "satan", "mscan", "saint", "worm"])

x_train = []
y_train = []
x_test = []

r= []
r2 = []
r3 = []
for data in all_train:
    if data[1] not in r:
        r.append(data[1])
    if data[2] not in r2:
        r2.append(data[2])
    if data[3] not in r3:
        r3.append(data[3])

print("Making x_trian, y_train")
for data in all_train:
    x = [0] * 3
    x[0] = r.index(data[1])
    x[1] = r2.index(data[2])
    x[2] = r3.index(data[3])
    x_train.append(x)

    y = [0, 0, 0, 0, 0]
    d = data[-1][:-1]
    if d == "normal":
        y[0] = 1
    elif d in dos:
        y[1] = 1
    elif d in u2r:
        y[2] = 1
    elif d in r2l:
        y[3] = 1
    elif d in probe:
        y[4] = 1
    y_train.append(y)

x_train = np.array(x_train)
y_train = np.array(y_train)

print(r)
print(r2)
print(r3)
print("x_train shape: ", x_train.shape)
print("y_train[0]", y_train[0])

# define model
model = Sequential()

elu = ELU(alpha = 0.1)

model.add(Dense(input_shape = x_train[0].shape, output_dim = 32))
model.add(elu)
model.add(Dropout(0.25))
model.add(Dense(output_dim = 32))
model.add(elu)
model.add(Dropout(0.25))
model.add(Dense(output_dim = 32))
model.add(elu)
model.add(Dropout(0.25))
model.add(Dense(output_dim = 32))
model.add(elu)
model.add(Dropout(0.25))
model.add(Dense(output_dim = 5))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
model.fit(x_train, y_train,
          batch_size = 4096, nb_epoch = 15, shuffle = True)



# predict and output

all_test = []
print("Loading test.in")
with open("../data/test.in", 'r') as f:
    for line in f:
        line = line.strip()
        all_test.append(line.split(','))
for data in all_test:
    if data[2] == "icmp":
        print(data)

print("Making x_test")
for data in all_test:
    x = [0] * 3
    x[0] = r.index(data[1])
    if data[2] in r2:
        x[1] = r2.index(data[2])
    else:
        x[1] = 0
    x[2] = r3.index(data[3])
    x_test.append(x)

x_test = np.array(x_test)

result = model.predict(x_test)
print("Output file")
out = []
for i in range(len(result)):
    m, idx = 0, 0
    for j in range(len(result[i])):
        if result[i][j] > m:
            m = result[i][j]
            idx = j
    out.append(idx)
ofile = open("../output/out1.csv", 'w')
ofile.write("id,label\n")
for i in range(len(out)):
    ofile.write(str(i+1) + ',')
    ofile.write(str(out[i]) + '\n')

