import numpy as np 
import pandas as pd 
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from subprocess import check_output
from nltk.stem import WordNetLemmatizer
print(check_output(["ls", "../data"]).decode("utf8"))

bio = pd.read_csv("../data/biology.csv")
cook = pd.read_csv("../data/cooking.csv")
crypto = pd.read_csv("../data/crypto.csv")
diy = pd.read_csv("../data/diy.csv")
robot = pd.read_csv("../data/robotics.csv")
travel = pd.read_csv("../data/travel.csv")
sample_sub = pd.read_csv("../data/sample_submission.csv")
test = pd.read_csv("../data/test.csv")


all_dat = [bio,cook,crypto,diy,robot,travel]

swords1 = stopwords.words('english')

punctuations = string.punctuation

def title_clean(data):
    title = data.title
    title = title.apply(lambda x: x.lower())
    print('Remove Punctuations')
    # title = [' '.join(word.strip(punctuations) for word in i.split()) for i in title]
    title = title.apply(lambda x: re.sub(r'^\W+|\W+$',' ',x))
    title = title.apply(lambda i: ''.join(i.strip(punctuations))  )
    print('tokenize')
    title = title.apply(lambda x: word_tokenize(x))
    print('Remove stopwords')
    title = title.apply(lambda x: [i for i in x if i not in swords1 if len(i)>2])
    print('minor clean some wors')
    title = title.apply(lambda x: [i.split('/') for i in x] )
    title = title.apply(lambda x: [i for y in x for i in y])
    print('Lemmatizing')
    wordnet_lemmatizer = WordNetLemmatizer()
    title = title.apply(lambda x: [wordnet_lemmatizer.lemmatize(i,pos='v') for i in x])
    title = title.apply(lambda x: [i for i in x if len(i)>2])
    return(title)

test.title = title_clean(test)
test.head()
tags = test.title.apply(lambda x: nltk.pos_tag(x) )
tags = tags.apply(lambda x: [i[0] for i in x if i[1][0] in "N" ])
test["tags"] = tags.apply(lambda x: " ".join(x))
sub_dat = test.loc[:,["id","tags"]]
sample_sub.head()
sub_dat.to_csv("sub0.csv",index=False)
sub_dat.head()
