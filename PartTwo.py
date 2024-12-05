import pandas
import glob
import os
from pathlib import Path
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import nltk
from nltk.stem import WordNetLemmatizer


path = Path.cwd() / "p2-texts"

file = glob.glob(os.path.join(path,"*.csv"))

for sheet in file:
    df = pandas.read_csv(sheet)



'''part 2 a'''
#part2ai
df['party'] = df['party'].replace({'Labour (Co-op)':'Labour'}) 

#part2aii
df=df.dropna(subset='party') #help taken from chatGPT to drop null values
parties = set(df['party']) 
partycount={}
for partyname in parties:
    count = 0
    for partylist in df['party']:
        if str(partylist) == str(partyname):
            count +=1
    partycount[partyname]= count
del partycount['Speaker']
partycount = sorted(partycount.items(), key=lambda x:x[1], reverse = True)

updated_partycount=[partycount[0],partycount[1],partycount[2],partycount[3]]

top4party =[]
for party in updated_partycount:
    partyname = str(party[0])
    top4party.append(partyname)
        
df = df[df['party'].isin(top4party)]

#part2aiii
df = df[df.speech_class == 'Speech']

#part2aiv
charcount=[] 
for row in df['speech']:
    speechlength = len(row)
    charcount.append(speechlength)

df=df.assign(characters = charcount)
df = df.drop(df[df.characters < 1500].index)
df = df.drop(columns=['characters'])

dfshape = df.shape

print(dfshape)

'''part 2 b'''
corpus = df['speech']
labels = df['party']

vectoriser = TfidfVectorizer(max_features=4000, stop_words='english')
orig_matrix=vectoriser.fit_transform(corpus)
# print(vectoriser.get_feature_names_out())
# print(orig_matrix.shape)

x_train, x_test,y_train,y_test = train_test_split(orig_matrix,labels, random_state=99) 
#have chosen to use default train and test parameters which are train=0.75 and test=0.25
#if parameters need changing, then after: random state=99, type: ', test_size=(whatever proportion required)


'''part 2 c'''
RFclass=RandomForestClassifier(n_estimators=400)
RFclass.fit(x_train,y_train)
RFy_pred=RFclass.predict(x_test)
RFclassrep=classification_report(y_test,RFy_pred,zero_division=1)
print('Random forest classification report:','\n', RFclassrep)


svmclass=svm.LinearSVC()
svmclass.fit(x_train,y_train)
svmy_pred=svmclass.predict(x_test)
svmclassrep=classification_report(y_test,svmy_pred, zero_division=1)
print('SVM classification report:','\n', svmclassrep)

'''RFacc = 0.73, svm acc=0.80'''

RFf1score = f1_score(y_test, RFy_pred, average='macro')
print('Random forest F1 score:',"\n", RFf1score)
svmf1score = f1_score(y_test,svmy_pred, average='macro')
print('SVM F1 score:',"\n", svmf1score)

'''part 2 d'''
new_vectoriser = TfidfVectorizer(max_features=4000, stop_words='english',ngram_range=(1,3))
new_matrix=new_vectoriser.fit_transform(corpus)

a_train, a_test,b_train,b_test = train_test_split(new_matrix,labels, random_state=99)
#have chosen to use default train and test parameters which are train=0.75 and test=0.25
#if parameters need changing, then after: random state=99, type: ', test_size=(whatever proportion required)

RFclass=RandomForestClassifier(n_estimators=400)
RFclass.fit(a_train,b_train)
RFb_pred=RFclass.predict(a_test)
RFclassrep=classification_report(b_test,RFb_pred,zero_division=1)
print('Random forest classification report(updated vectoriser):','\n', RFclassrep)

svmclass=svm.LinearSVC()
svmclass.fit(a_train,b_train)
svmb_pred=svmclass.predict(a_test)
svmclassrep=classification_report(b_test,svmb_pred,zero_division=1)
print('SVM classification report(updated vectoriser):','\n', svmclassrep)

'''rf acc= 0.74, svm acc = 0.81'''

RFf1score = f1_score(b_test, RFb_pred, average='macro')
print('Random forest F1 score(updated vectoriser):',"\n", RFf1score)
svmf1score = f1_score(b_test,svmb_pred, average='macro')
print('SVM F1 score(updated vectoriser):',"\n", svmf1score)

'''part 2 e'''
def Lemma_tokenizer(text): #run through chatGPT to check if need for debugging
    lemmatizer=WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemma_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemma_tokens

custom_vectorizer = TfidfVectorizer(tokenizer=Lemma_tokenizer,max_features=4000, stop_words='english',ngram_range=(1,3))
custom_matrix=custom_vectorizer.fit_transform(corpus)

a_train, a_test,b_train,b_test = train_test_split(custom_matrix,labels, random_state=99)

RFclass=RandomForestClassifier(n_estimators=400)
RFclass.fit(a_train,b_train)
RFb_pred=RFclass.predict(a_test)
RFclassrep=classification_report(b_test,RFb_pred,zero_division=1)


svmclass=svm.LinearSVC()
svmclass.fit(a_train,b_train)
svmb_pred=svmclass.predict(a_test)
svmclassrep=classification_report(b_test,svmb_pred,zero_division=1)


RFf1score = f1_score(b_test, RFb_pred, average='macro')
svmf1score = f1_score(b_test,svmb_pred, average='macro')

RFweightedscore=f1_score(b_test,RFb_pred, average='weighted')
svmweightedscore=f1_score(b_test,svmb_pred, average='weighted')

if RFweightedscore > svmweightedscore and RFf1score > svmf1score:
    print('Best performance for custom tokenizer: Random forest classification report:','\n', RFclassrep)
elif RFweightedscore < svmweightedscore and RFf1score < svmf1score:
    print('Best performance for custom tokenizer: SVM classification report:','\n', svmclassrep)
else:
    print('Random forest classification report(custom vectoriser):','\n', RFclassrep)
    print('SVM classification report(custom vectoriser):','\n', svmclassrep)