##This has been made by M Dhruv and Pabitra Sharma for NNLS final term project

#Import Statements
import pandas as pd
import numpy as np
import re
from sklearn import svm
data = pd.read_csv('database_UCI.csv',encoding = 'unicode_escape',keep_default_na=False)
#Making Inputs
lst = []
unwanted = ['=','http','@','ly','__','--','|','_','.)','..','www','com','<','>','[',']','{','}','0','1','2','3','4','5','6','7','8','9']
bs = ['hi','thanks','respected','dear','sir','a','so','then', 'at', 'is', 'further', 'to', 'too', 'or', 'as', 'if', 'for', 'by', 'this', 'that', 'thus', 'when', 'while', 'and', 'yet', 'or', 'nor', 'finally', 'also', 'besides', 'addition', 'moreover', 'previously', 'meanwhile']
word_list = []
# First we create the dictionary of words
for i in range(500):
    lst.append(i)
np.random.shuffle(lst)
for i in lst:
    for j in range(2):
        #print(i)
        para = data.iloc[i,j]
        words = re.split(r"[\s \n - _ / . ? , ! \' \" : ;]",para)
        for word in words:
            word = word.lower() # we make all the words lowercase
            word = word.replace('(','')
            word = word.replace(')','')
            word = word.replace('*','')
            word = word.replace('#','')
            check = 0
            for crap in unwanted:
                if crap in word:
                    check = 1
            if (check == 0) and (word not in bs) and (len(word)<20) and  (len(word)>1):
               word_list.append(word)

# this is for removing duplicates
final_word_list = []
for word in word_list:
    if word not in final_word_list:
        final_word_list.append(word)
#print(final_word_list)
#print(len(final_word_list))
# This completes the generation of the dictionary

#Now we implement the SVM
#Inputs
input = np.zeros((500,len(final_word_list)))
for i in range(500):
    words_in_mail = []
    for j in range(2):
        para = data.iloc[i,j]
        words = re.split(r"[\s \n - _ / . ? , ! \' \" : ;]",para)
        for word in words:
            word = word.lower() # we make all the words lowercase
            word = word.replace('(','')
            word = word.replace(')','')
            word = word.replace('*','')
            word = word.replace('#','')
            check = 0
            for crap in unwanted:
                if crap in word:
                    check = 1
            if (check == 0) and (word not in bs) and (len(word)<20) and  (len(word)>1):
               words_in_mail.append(word)
    for k in range(len(final_word_list)):
        count = 0
        for word in words_in_mail:
            if word == final_word_list[k]:
                count = count + 1
        input[i][k] = count
#print(final_word_list)
output = []
for i in range(500):
    output.append(data.iloc[i][0])
#print(input[49][200])
clf = svm.SVC()
clf.fit(input, output)
arr = clf.support_vectors_

# Now to run the demo
demo_input = np.zeros(len(final_word_list))
words_in_mail = []
file = open("demo_message.txt", "r")
para = file.read()
file.close()
#print(para)
words = re.split(r"[\s \n - _ / . ? , ! \' \" : ;]",para)
for word in words:
    word = word.lower() # we make all the words lowercase
    word = word.replace('(','')
    word = word.replace(')','')
    word = word.replace('*','')
    word = word.replace('#','')
    check = 0
    for crap in unwanted:
        if crap in word:
            check = 1
    if (check == 0) and (word not in bs) and (len(word)<20) and  (len(word)>1):
       words_in_mail.append(word)
#print(words_in_mail)
for k in range(len(final_word_list)):
    count = 0
    for word in words_in_mail:
        if word == final_word_list[k]:
            count = count + 1
    demo_input[k] = count
print(demo_input)
print(clf.predict([demo_input]))
