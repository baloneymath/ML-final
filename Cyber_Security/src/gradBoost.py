#! /usr/bin/env python3

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier

# Building the attack type map
atk_type = {'normal': 0, 'dos': 1, 'u2r': 2, 'r2l': 3, 'probe': 4}
atk_file = open("../data/training_attack_types.txt", 'r')

for line in atk_file:
    ss = line.split()
    atk_type[ ss[0] ] = atk_type[ ss[1].split('\n')[0] ]

print("Attack type mapping : ", atk_type)
print()

# Read data
train_data = open("../data/train", 'r')
test_data = open("../data/test.in", 'r')
x = []
y = []

protocols = {}
services = {}
flags = {}

print("Reading Files...")
for line in train_data:
    s = line.split(',')
    y.append( atk_type[ s[-1][:-2] ] )

    # For column 2, 3, 4 need map to integer.
    if s[1] in protocols:
        s[1] = protocols[ s[1] ]
    else:
        protocols[ s[1] ] = len(protocols)
        s[1] = protocols[ s[1] ]

    if s[2] in services:
        s[2] = services[ s[2] ]
    else:
        services[ s[2] ] = len(services)
        s[2] = services[ s[2] ]

    if s[3] in flags:
        s[3] = flags[ s[3] ]
    else:
        flags[ s[3] ] = len(flags)
        s[3] = flags[ s[3] ]

    x.append(list(map(float, s[:-1])))

X = np.array(x)

print("Start training...")
boost = OneVsRestClassifier(GradientBoostingClassifier(n_estimators = 700, criterion = "friedman_mse",
                                learning_rate = 0.015, loss = "deviance",
                                min_impurity_split = 1e-7, max_depth = 3,
                                #subsample = 0.5, max_features = 'sqrt',
                                verbose = 1), n_jobs = -1)
boost.fit(X, y)

print("Start fitting...")

x_test = []
for line in test_data:
    s = line.split(',')
    if s[1] in protocols:
        s[1] = protocols[ s[1] ]
    else:
        protocols[ s[1] ] = len(protocols)
        s[1] = protocols[ s[1] ]

    if s[2] in services:
        s[2] = services[ s[2] ]
    else:
        services[ s[2] ] = len(services)
        s[2] = services[ s[2] ]

    if s[3] in flags:
        s[3] = flags[ s[3] ]
    else:
        flags[ s[3] ] = len(flags)
        s[3] = flags[ s[3] ]

    x_test.append(list(map(float, s)))

X_test = np.array(x_test)

result = boost.predict(X_test)

with open("../output/gradBoost.csv", 'w') as f:
    f.write("id,label\n")
    for i in range(len(result)):
        f.write("{0},{1}\n".format(i+1, result[i]))



