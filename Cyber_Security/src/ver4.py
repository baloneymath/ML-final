#!/usr/bin/env python3

import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
Y = np.array(y)

print("Start training...")
classifiers = []
for t in range(5):
    forest = RandomForestClassifier(verbose = 1, n_estimators = 50, n_jobs = 8)
#n_estimators = 256, criterion = "gini",
#                                max_features = "sqrt", n_jobs = 8,
#                                min_impurity_split = 1e-7,
#                                verbose = 1)
    forest.fit(X, (Y == t).astype(int))
    classifiers.append(forest)
    print('Finished', t)

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

results = []
for i in range(5):
    print(classifiers[i].classes_)
    r = classifiers[i].predict_proba(X_test)[:,1]  # Get second column only
    results.append(r)

# Weight tuning
results[0] = results[0] * 0.5
results[2] = results[2] * 10
results[3] = (results[3] > 0).astype(int).astype(float)

for r in results:
    r = r.tolist()

result = np.array(results).argmax(axis=0)

zeros = 0
ones = 0
twos = 0
threes = 0
fours = 0

with open("../output/out44.csv", 'w') as f:
    f.write("id,label\n")
    for i in range(len(result)):
        f.write("{0},{1}\n".format(i+1, result[i]))
        if result[i] == 0:
            zeros += 1
        elif result[i] == 1:
            ones += 1
        elif result[i] == 2:
            twos += 1
        elif result[i] == 3:
            threes += 1
        elif result[i] == 4:
            fours += 1

print('0\'s: {}'.format(zeros))
print('1\'s: {}'.format(ones))
print('2\'s: {}'.format(twos))
print('3\'s: {}'.format(threes))
print('4\'s: {}'.format(fours))


