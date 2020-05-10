import spacy
nlp = spacy.load("en_core_web_sm")
nlp2 = spacy.load("en_core_web_lg")
import pandas as pd
import numpy as np
import glob #finds all the pathnames matching a specified pattern
from tqdm import tqdm #Instantly makes your loops show a smart progress meter

### Import Method 

from nltk.corpus import stopwords
import re
from textblob import TextBlob

path = r'C:/Users/DELL/Desktop/innodatatics/datas' # use your path
all_files = glob.glob(path + "/*.xlsx")

li = []

for filename in all_files:
    df = pd.read_excel(filename, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame.columns =['sr_no', 'bus_desc', 'verdict', 'reasoning'] #renaming columns

frame = frame.iloc[4:,:] #dropping initial N/A columns 

#dropping remaining columns with N/A values
frame = pd.DataFrame.dropna(frame, axis=0, how='any', thresh=None, subset=None, inplace=False) 

#dropping rows that have the initial column names repeated
frame = frame[frame.sr_no != 'Sr. No.']

#building association vectors for en_core corpora
#define all vectors and strings
strings = []
vectors = []
for key, vector in tqdm(nlp2.vocab.vectors.items(), total = len(nlp2.vocab.vectors.keys())):
    try:
        strings.append(nlp2.vocab.strings[key])
        vectors.append(vector)
    except:
        pass

vectors = np.vstack(vectors)

## counting words
frame['word_count'] = frame['bus_desc'].apply(lambda x: len(str(x).split(" ")))


## COunting Charater ++ Including white Space 

frame['char_count'] = frame['bus_desc'].str.len()


def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

frame['avg_word'] = frame['bus_desc'].apply(lambda x: avg_word(x))
frame[['bus_desc','avg_word']].head()


### STop Words 

stop = stopwords.words('english')


frame['stopwords'] = frame['bus_desc'].apply(lambda x: len([x for x in x.split() if x in stop]))


##Special Charater 
#frame['hastags'] = frame['bus_desc'].apply(lambda x: len([x for x in x.split() if x.startswith('#')])



##Number of numerics

frame['numerics'] = frame['bus_desc'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))



## Upper Case 
frame['upper'] = frame['bus_desc'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
frame[['bus_desc','upper']].head()

#PreProcessing 


## To Lower 
frame['bus_desc'] = frame['bus_desc'].apply(lambda x: " ".join(x.lower() for x in x.split()))

## Special Puncuation
frame['bus_desc'] = frame['bus_desc'].str.replace('[^\w\s]','')


## Removing StopWords
stop = stopwords.words('english')
frame['bus_desc'] = frame['bus_desc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

## Remove Common Word 
freq = pd.Series(' '.join(frame['bus_desc']).split()).value_counts()[:15]

freq = list(freq.index)

frame['bus_desc'] = frame['bus_desc'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

## Rare word removal
freq = pd.Series(' '.join(frame['bus_desc']).split()).value_counts()[-15:]

freq = list(freq.index)
frame['bus_desc'] = frame['bus_desc'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
frame['bus_desc'].head()


## Spelling correction

frame['bus_desc'].apply(lambda x: str(TextBlob(x).correct()))

## Tokenization

frame['bus_desc'].reset_index(drop=True, inplace=True)


text = frame['bus_desc'].astype(str)

print(text)

TextBlob(text).words


























