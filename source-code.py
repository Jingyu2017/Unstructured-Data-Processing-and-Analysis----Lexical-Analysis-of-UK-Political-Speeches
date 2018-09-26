#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Coursework2
Part 1

Mar 2018
@author: Jingyu Li 
k1756990
"""

import sys
import io
import math
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
import re
import glob
import operator
DEBUGGING=False

#returns a list of path names that match pathname "../speeches/*.txt"
filenames = glob.glob("../speeches/*.txt")
numFiles = len(filenames)
df = pd.DataFrame(columns=list('ABCDEFG'))  
df['A']= [filenames[i].split('/')[-1] for i in range(numFiles)]

# Sentiment analysis (before stopwords being removed)
polarity=[]
subjectivity=[]
for i in range(numFiles): 
    try:
        with io.open(filenames[i],'r',encoding='utf-8') as f:
            txt=f.read()
            blob = TextBlob(txt)        
            polarity.append(blob.sentiment.polarity)
            subjectivity.append( blob.sentiment.subjectivity)
    except Exception as x:
        print 'An error occured when reading the speeches docs' + str( x )
        sys.exit()    
df['B']= polarity
df['C']= subjectivity

#read in stopwords 
try:
    with io.open('../english-stop-words-large.txt', 'r',encoding='utf-8') as f:
        stopwords = f.read().splitlines()
except Exception as x:
    print 'An error occured when reading the data file:' + str( x )
    sys.exit()
    
#Use regular expression to express every pattern to be substituted and its corresponding substitution
#in utf-8 coding single quote is coded with “\u2019”
replacement_patterns = [
	(u'won(\u2019)t', 'will not'),
	(u'can(\u2019)t', 'cannot'),
	(u'i(\u2019)m', 'i am'),
	(u'ain\'t', 'is not'),
	(u'(\w+)(\u2019)ll', '\g<1> will'),
	(u'(\w+)(\u2019)t', '\g<1> not'),
	(u'(\w+)(\u2019)ve', '\g<1> have'),
	(u'(\w+)(\u2019)s', '\g<1> is'),
	(u'(\w+)(\u2019)re', '\g<1> are'),
	(u'(\w+)(\u2019)d', '\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in replacement_patterns]

#attributes D-I are based on lists of words which have been removed of 
#stop-words and non-alphabetic terms
word_count=[]
most_common_word =[]
freq=[]
norm_freq=[]
collections=[]
word_set=set()
for i in range(numFiles): 
    try:
        with io.open(filenames[i],'r',encoding='utf-8') as f:
            txt=f.read()
            #substitute some instances with its corresponding substitution pattern
            for (pattern, repl) in patterns:
                txt = re.sub(pattern, repl, txt)
            #append each speech to a list as a string
            collections.append(txt.lower()) 
            
            tokens = nltk.word_tokenize(txt)
            
            words=[w.lower() for w in tokens if w.lower() not in stopwords and w.isalpha()]            
            
            #save a set of words for the later implementation of term-document frequency matrix
            s = set(words)
            word_set = word_set.union(s) 

            word_count.append(len(words))
            fdist = nltk.FreqDist(words)
            freq.append(fdist.most_common(1)[0][1])
            most_common_word.append(fdist.most_common(1)[0][0])
            
            norm_freq.append( 1.*freq[-1]/word_count[-1])
    except Exception as x:
        print filenames[i]
        print 'An error occured when reading the speeches docs:' + str( x )
        sys.exit() 
        
df['D']= word_count       
df['E']= most_common_word
df['F']= norm_freq
df['G']= freq
idf=[]
for i in range(numFiles): 
    w=most_common_word[i]
    appear = [(w in j) for j in collections] #collections is a list of speech strings
    #calculate how many documents in the collections present the most common word
    numAppear = sum(appear)
    # calsulate inverse document frequency
    idf.append(math.log(1.0*numFiles/(1+numAppear)))
df['H']= idf
df['I']= df['G']*df['H']


if DEBUGGING:
    print 'This is the table for main characteristics extracted from each speech'
    print df
    
    
# =============================================================================
# generate term_document_matrix
# =============================================================================
#sort the word set in alphabetical order
terms=sorted(list(word_set)) 
term_docu_matrix = pd.DataFrame(index=terms)
for i in range(numFiles):
    tokens = nltk.word_tokenize(collections[i])
#column: document index
#index : words
    term_docu_matrix[str(i)]= [tokens.count(j) for j in terms]

if DEBUGGING:
    print 'shape of term document matrix:'
    print term_docu_matrix.shape


# =============================================================================
# compute the pairwise distance between documents
# identify two closest documents 
# =============================================================================
def cosine (x,y):
    return 1.* np.dot(x,y) / (math.sqrt(np.dot(x,x)) * math.sqrt(np.dot(y,y)))
#The pairwise distance between all pairs of documents is stored in a dictionary 
#where the key is a tuple referring the indices if two documents
cos_distance={}
for i in range(numFiles): 
    for j in range(numFiles): 
        if i<j:
            x=term_docu_matrix.iloc[i]
            y=term_docu_matrix.iloc[j]
            cos_distance[(i,j)]=cosine (x,y)
        
#sort the dictionary by value, in descending order
 #this returns a list
cos_distance = sorted(cos_distance.items(), key=operator.itemgetter(1), reverse=True)           

#the first [0] is there to extract []
#the secons [0] is there to extract the documents pair 
doc1=cos_distance[0][0][0]
doc2=cos_distance[0][0][1]     
cos_similarity=cos_distance[0][1]    
if DEBUGGING:
    print '%s and %s are the closest with a cosine distance of %.4f'%(df['A'][doc1],df['A'][doc2],cos_similarity)          
     

# =============================================================================
#  COMPARATIVE ANALYSIS          
# =============================================================================
corbyn_index = [i for i in range(numFiles) if df['A'][i].startswith("corbyn-2017") ] 
corbyn_allSpeeches = ' '.join([collections[i] for i in corbyn_index])
corbyn_words = nltk.word_tokenize(corbyn_allSpeeches)
corbyn2017=[wd for wd in corbyn_words if wd not in stopwords and wd.isalpha()]


may_index = [i for i in range(numFiles) if df['A'][i].startswith("may-2017") ] 
may_allSpeeches = ' '.join([collections[i] for i in may_index])        
may_words = nltk.word_tokenize(may_allSpeeches)
may2017=[wd for wd in may_words if wd not in stopwords and wd.isalpha()]
    
may_top_10_bigram = nltk.FreqDist(nltk.bigrams(may2017)).most_common(10)  
corbyn_top_10_bigram = nltk.FreqDist(nltk.bigrams(corbyn2017)).most_common(10)

may_top_10 = nltk.FreqDist(may_words).most_common(10) 
corbyn_top_10 = nltk.FreqDist(corbyn_words).most_common(10)  

if DEBUGGING:
    nltk.FreqDist(nltk.bigrams(may2017)).plot(10)
    nltk.FreqDist(nltk.bigrams(corbyn2017)).plot(10)
    
  
index = [i for i in range(numFiles) if df['A'][i].startswith("corbyn") or df['A'][i].startswith("may") ] 
allSpeeches = ' '.join([collections[i] for i in index])
words = nltk.word_tokenize(allSpeeches)
UKcorpus=[wd for wd in words if wd not in stopwords and wd.isalpha()]

def norm_frequency(wordsList):
    
    fd = nltk.FreqDist(wordsList)
    total = fd.N()
    for word in fd:
        fd[word] /= float(total)  #normalised frequency of most frequent word
    p=pd.Series(fd) 
    frequency_df=pd.Series.to_frame(p)
    #frequency_df.columns = [str(wordsList)]
    return frequency_df

uk_df = norm_frequency(UKcorpus)
corbyn_df = norm_frequency(corbyn2017)

uk_df.columns=["uk"]
corbyn_df.columns=["corbyn"]
t=pd.merge(corbyn_df, uk_df, left_index=True, right_index=True, how='left')
t['diff']=t.corbyn-t.uk
if DEBUGGING:
    print t.sort_values(by=['diff'], ascending=False)

may_df = norm_frequency(may2017)
may_df.columns=["may"]

t2=pd.merge(may_df, uk_df, left_index=True, right_index=True, how='left')
t2['diff']=t2.may-t2.uk
if DEBUGGING:
    print t2.sort_values(by=['diff'], ascending=False)

# =============================================================================
#  ASSOCIATION RULES           
# =============================================================================
            
corbyn_index = [i for i in range(numFiles) if df['A'][i].startswith("corbyn") ]
term_docu_matrix_cb = term_docu_matrix.iloc[:,corbyn_index]
corbyn_top3 = term_docu_matrix_cb.sum(axis=1).sort_values(ascending=False)[:3]

print 'corbyn’s 3 most frequently used words and their frequency:'
print corbyn_top3
print 'corbyn’s 3 most frequently used words and the number of speeches it is used in:'
print term_docu_matrix_cb.astype(bool).sum(axis=1)[corbyn_top3.index]

assoc_matrix=term_docu_matrix_cb.astype(bool).astype(int).loc[[u'people', u'labour', u'party']]

def coverage(word1_index,word2_index):
    rule_valid = 0
    for i in range(assoc_matrix.shape[1]):
        if assoc_matrix.iloc[word1_index,i] == 1:  
            if assoc_matrix.iloc[word2_index,i] == 1:                
                rule_valid += 1        
    return rule_valid

print 'coverage(A ⇒ B) ',coverage(0,1)
print 'coverage(A ⇒ C) ',coverage(0,2)
print 'coverage(B ⇒ A) ',coverage(1,0)
print 'coverage(B ⇒ C) ',coverage(1,2)
print 'coverage(C ⇒ A) ',coverage(2,0)
print 'coverage(C ⇒ B) ',coverage(2,1)


def accuracy(LHS,RHS):
    #LHS:left-hand side,RHS:right-hand side
    return 1.*coverage(LHS,RHS)/assoc_matrix.iloc[LHS,:].sum()

print 'accuracy(A ⇒ B) ',accuracy(0,1)
print 'accuracy(A ⇒ C) ',accuracy(0,2)
print 'accuracy(B ⇒ A) ',accuracy(1,0)
print 'accuracy(B ⇒ C) ',accuracy(1,2)
print 'accuracy(C ⇒ A) ',accuracy(2,0)
print 'accuracy(C ⇒ B) ',accuracy(2,1)


