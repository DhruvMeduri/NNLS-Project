import pandas as pd
import numpy as np
data = pd.read_csv('NNLS_data.csv',encoding = 'unicode_escape',keep_default_na=False)
#data.fillna('Dummy')
lst = []
dict = {}
count = -1
unwanted = ['=','http','@','ly','__','--','|','_','\0','\1','\2','\3','\4','\5','\6','\7','\8','\9','.)','..','www','<','>','[',']','{','}','0','1','2','3','4','5','6','7','8','9']
bs = ['hi','thanks','respected','dear','sir','a','so','then', 'at', 'is', 'further', 'to', 'too', 'or', 'as', 'if', 'for', 'by', 'this', 'that', 'thus', 'when', 'while', 'and', 'yet', 'or', 'nor', 'finally', 'also', 'besides', 'addition', 'moreover', 'previously', 'meanwhile']
word_list = []
# First we create the dictionary of words
for i in range(4232):
    lst.append(i)
np.random.shuffle(lst)
for i in lst:
    for j in range(2):
        #print(i)
        para = data.iloc[i,j]
        words = para.split()
        for word in words:
            word = word.lower() # we make all the words lowercase
            word = word.replace('.','')
            word = word.replace('.','!')
            word = word.replace(',','')
            word = word.replace(':','')
            word = word.replace(';','')
            word = word.replace('(','')
            word = word.replace(')','')
            word = word.replace('!','')
            word = word.replace('?','')
            word = word.replace('*','')
            word = word.replace('\'','')
            word = word.replace('\"','')
            word = word.replace('#','')
            #word = word.replace(''','')
            #word = word.replace('"','')
            check = 0
            for crap in unwanted:
                if crap in word:
                    check = 1
            if (check == 0) and (word not in bs):
               word_list.append(word)
               #count = count + 1

# this is for removing duplicates
final_word_list = []
for word in word_list:
    if word not in final_word_list:
        final_word_list.append(word)
print(final_word_list)
print(len(final_word_list))

#print(dict)
