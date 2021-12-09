##This has been made by M Dhruv and Pabitra Sharma for NNLS final term project

import pandas as pd
import numpy as np
import re
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import sklearn.metrics as matrices
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('NNLS_data.csv',encoding = 'unicode_escape',keep_default_na=False)


random_input = np.random.permutation(4000)
#print(random_input)


#data.fillna('Dummy')
lst = []
unwanted = ['=','http','@','ly','__','--','|','_','.)','..','www','com','<','>','[',']','{','}','0','1','2','3','4','5','6','7','8','9']
bs = ['hi','thanks','respected','dear','sir','a','so','then', 'at', 'is', 'further', 'to', 'too', 'or', 'as', 'if', 'for', 'by', 'this', 'that', 'thus', 'when', 'while', 'and', 'yet', 'or', 'nor', 'finally', 'also', 'besides', 'addition', 'moreover', 'previously', 'meanwhile']
word_list = []
# First we create the dictionary of words
for i in range(4000):
    lst.append(i)
np.random.shuffle(lst)
for i in lst:
    for j in range(2):
        #print(i)
        para = data.iloc[random_input[i],j]
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
input = np.zeros((3800,len(final_word_list)))
for i in range(3800):
    words_in_mail = []
    for j in range(3):
        para = data.iloc[random_input[i],j]
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
input_test = np.zeros((200,len(final_word_list)))
for i in range(3801, 4000):
    words_in_mail = []
    for j in range(3):
        para = data.iloc[random_input[i],j]
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
        input_test[i-3801][k] = count


#print(input_test)


output = []
for i in range(3800):
    output.append(data.iloc[random_input[i]][3])

y = np.array([(1 if i=='SPAM' else 0) for i in output])
#print(y)
output_test = []
for i in range(3800, 4000):
    output_test.append(data.iloc[random_input[i]][3])


y_test = np.array([(1 if i=='SPAM' else 0) for i in output_test])

#print(output)
clf = svm.SVC(C = 200000, probability=True)

clf.fit(input, y)
arr = clf.support_vectors_


predict = clf.predict(input_test)



clf2 = MultinomialNB(alpha=15.0)
clf2.fit(input, y)
predict2 = clf2.predict(input_test)

'''
mis = 0

for i in range(50):
    if predict[i] != output_test[i]:
        mis += 1

mis2 = 0

for i in range(50):
    if predict2[i] != output_test[i]:
        mis2 += 1

'''
output_prob = clf.predict_proba(input_test)

test = np.zeros(len(output_test))
pred = np.zeros(len(predict))

for i in range(len(output_test)):
    if output_test[i] == 'spam':
        test[i] = 1

for i in range(len(predict)):
    if predict[i] == 'spam':
        pred[i] = 1


'''
Accuracy = (mis*100)/50

Accuracy2 = (mis2*100)/50
'''

'''
y_score = clf.predict_proba(input_test)
y_score = np.array(y_score)

y_score2 = clf2.predict_proba(input_test)
y_score2 = np.array(y_score)
#print(y_score)

from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, neg_label=0, pos_label=1, classes=[0,1])
y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

#print(y_test_bin)
#print(arr)
#print(len(arr))
# This is to predict


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr = dict()
tpr = dict()
roc_auc = dict()


fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()

for i in [0, 1]:
    # collect labels and scores for the current index
    labels = y_test_bin[:, i]
    scores = y_score[:, i]

    labels2 = y_test_bin[:, i]
    scores2 = y_score2[:, i]

    # calculates FPR and TPR for a number of thresholds
    fpr[i], tpr[i], thresholds = matrices.roc_curve(labels, scores)
    fpr2[i], tpr2[i], thresholds2 = matrices.roc_curve(labels2, scores2)

    # given points on a curve, this calculates the area under it
    roc_auc[i] = auc(fpr[i], tpr[i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])

fpr["micro"], tpr["micro"], _ = matrices.roc_curve(y_test_bin.ravel(), y_score2.ravel())
roc_auc['micro'] = auc(fpr["micro"], tpr["micro"])

fpr2["micro"], tpr2["micro"], _ = matrices.roc_curve(y_test_bin.ravel(), y_score2.ravel())
roc_auc2['micro'] = auc(fpr2["micro"], tpr2["micro"])


plt.figure()
lw = 2

plt.plot(fpr2[1], tpr2[1], color='blue', lw=lw,)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
'''

matrices.plot_confusion_matrix(clf, input_test, y_test)
plt.show()

matrices.plot_confusion_matrix(clf2, input_test, y_test)
plt.show()