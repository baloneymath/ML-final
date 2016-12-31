import numpy as np
import pandas as pd

vector = np.zeros((2999335, 397), np.float)

topic_dtypes = {'document_id': np.int32, 'topic_id': np.int32, 'confidence_level': np.float}
topics = pd.read_csv('../data/documents_topics.csv', usecols = ['document_id', 'topic_id', 'confidence_level'], dtype=topic_dtypes)

# arr_topic = topics.as_matrix()
arr_documents = topics['document_id']
arr_topics = topics['topic_id']
arr_conf = topics['confidence_level']

del topics
# for topic in arr_topic:
#     vector[topic['document_id']][topic['topic_id']] = topic['confidence_level']
# del arr_topic

for index in range(len(arr_documents)):
    vector[arr_documents[index]][arr_topics[index]] = arr_conf[index]

del arr_documents
del arr_topics
del arr_conf

print('Finish topics.')

category_dtypes = {'document_id': np.int32, 'category_id': np.int32, 'confidence_level': np.float}
categories = pd.read_csv('../data/documents_categories.csv', usecols = ['document_id', 'category_id', 'confidence_level'], dtype=category_dtypes)

cateId_mapping = np.zeros((2101, ), np.int)
index = 300
prev = -1

cate_ids = categories['category_id']
cate_ids = sorted(cate_ids)
for temp_id in cate_ids:
    if (temp_id != prev):
        prev = temp_id
        cateId_mapping[prev] = index
        index += 1

# arr_cate = categories.as_matrix()
arr_documents = categories['document_id']
arr_cates = categories['category_id']
arr_conf = categories['confidence_level']

del categories
# for cate in arr_cate:
#     vector[cate['document_id']][cateId_mapping[cate['category_id']]] = cate['confidence_level']
# del arr_cate

for index in range(len(arr_documents)):
    vector[arr_documents[index]][cateId_mapping[arr_cates[index]]] = arr_conf[index]

del arr_documents
del arr_cates
del arr_conf

print('Finish categories.')

# caddtegories.sort_values(['document_id', 'confidence_level'])
# topics.sort_values(['document_id', 'confidence_level'])

# events_dtypes = {'display_id': np.int32, 'document_id': np.int32}
# events = pd.read_csv('../data/events.csv', usecols = ['display_id', 'document_id'], dtype = events_dtypes)
events_dtypes = {'document_id': np.int32}
events = pd.read_csv('../data/events.csv', usecols = ['document_id'], dtype = events_dtypes)
displayToDocument = events.as_matrix()
del events
displayToDocument = np.resize(displayToDocument, (len(displayToDocument), ))
displayToDocument = np.insert(displayToDocument, 0, 0)

print('Finish displayToDocument.')

train_dtypes = {'display_id': np.int32, 'ad_id': np.int32, 'clicked': np.int8}
train = pd.read_csv("../data/clicks_train.csv", usecols = ['display_id', 'ad_id', 'clicked'], dtype=train_dtypes)
arr_train = train.as_matrix()
del train

ad_size = 567074
ad_prop = np.zeros((ad_size, 397), np.float)
selected_count = np.zeros((ad_size, ), np.float)

index = 0
train_size = len(arr_train)

print('Start calculate trains.')
step = 1
level = train_size / 10

while (index < train_size):
    if index >= level:
        print('percentage: ', step, '0%')
        step += 1
        level = step * train_size / 10

    display_id = arr_train[index][0]
    document_id = displayToDocument[display_id]
    top = index + 1
    while (top < train_size and display_id == arr_train[top][0]):
        top += 1

    delta = 1.0 / (top - index - 1)
    
    while (index != top):
        ad_id = arr_train[index][1]
        if (arr_train[index][2] == 1):
            selected_count[ad_id] += 1
            ad_prop[ad_id] += vector[document_id]
        else:
            ad_prop[ad_id] -= delta * vector[document_id]

        index += 1

del arr_train

print('Finish trains.')

def getLikelihood(display_id, ad_id):
    document_id = displayToDocument[display_id]
    product = np.dot(vector[document_id], ad_prop[ad_id])
    product += 0.1 * selected_count[ad_id]
    return product

test_dtypes = {'display_id': np.int32, 'ad_id': np.int32}
test = pd.read_csv("../data/clicks_test.csv", usecols = ['display_id', 'ad_id'], dtype=test_dtypes)
# test['likelihood'] = getLikelihood(test['display_id'], test['ad_id'])
test['likelihood'] = test.apply(lambda row: getLikelihood(row['display_id'], row['ad_id']), axis=1)
test.sort_values(['display_id', 'likelihood'], inplace = True, ascending = False)
subm = test.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str, x)))
subm = subm.reset_index()
subm.to_csv("../output/test1.csv", index = False)

