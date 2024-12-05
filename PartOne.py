import nltk
import spacy
from pathlib import Path
import pandas as pd
import os
import glob
import re
from nltk.corpus import cmudict 
import en_core_web_sm
import itertools
import math


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


def count_syl(word, d): # debugged through ChatGPT
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters
    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.
    Returns:
        int: The number of syllables in the word.
    """
    syllablecount= 0
    lowerword = word.lower()
    phoneword = d.get(lowerword)
    if phoneword: #following lines until return were run through chatgpt for debugging, and adapted code suggested after debugging
        for phone in phoneword:
            for ph in phone:
                if ph[-1].isdigit():
                    syllablecount+=1
    return(syllablecount)
    pass

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.
    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.
    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sentences = nltk.sent_tokenize(text)
    sentcounts = len(sentences) 

    words = nltk.word_tokenize(text)
    wordcount = len(words)

    syllablecount = 0
    for word in words:
        x = count_syl(word, d)
        syllablecount += x
    
    #fkreadscore = 206.835 -(1.015*(wordcount/sentcounts))-(84.6*(syllablecount/wordcount))  #hashed out as was original formula used, but moodle update indicated to use following formula below
    fkreadlevel = (0.39*(wordcount/sentcounts))+(11.8*(syllablecount/wordcount))-15.59
    return(fkreadlevel)
    pass

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    files = os.scandir(path)
    story = glob.glob(os.path.join(path,"*.txt"))

    texts = []
    for file in files:
        if file.is_file():
            filename =(file.name)
            texts.append(filename)
    texts = [file.replace(".txt", "") for file in texts]
    texts = [str(text) for text in texts]

    fulltexts = []
    for file in story:
        storytext=open(file,'r')
        fulltexts.append(storytext.read())
    finaltexts=[]
    for text in fulltexts:
        text = text.replace('\n',' ')
        finaltexts.append(text)

    splittexts = [text.split("-") for text in texts]
    titles= []
    for title in splittexts:
        titles.append(title[0])
    author = []
    for name in splittexts:
        author.append(name[1])
    year = []
    for date in splittexts:
        year.append(date[2])
    
    data = {"text":finaltexts,"title":titles,"author":author,"year":year}
    df = pd.DataFrame(data)
    finaldf = df.sort_values("year")
    return(finaldf)
    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    #run through chat gpt to help debug
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 3000000
    
    texts = df['text'].tolist()

    parsed=[]
    for i in texts:
        x=nlp(i)
        parsed.append(x)
    df['parsed'] = parsed
    
    df.to_pickle(f'{store_path}/{out_name}' )
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    text= re.sub(r'[^\w]', ' ', text)
    text= text.lower()
    tokens = nltk.word_tokenize(text)
    types = set(tokens)
    ttr_value = (len(types)/len(tokens))
    return(ttr_value)
    pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    '''probabilities used in this are the probability of each bigram out of the set of bigrams (verb 'to say', its subject)
    probability of each verb out of the total number of verbs 'to say'
    probability of each syntactic subject out of the total number of syntactic subjects
    unsure if the probabilities are the correct ones to use, but PMI has been calculated based off these values'''
    doc_nountokenlist=[]
    doc_verb_list=[]
    doc_bigrams_list=[]
    for word in doc:
        if word.lemma_ == target_verb and word.pos_ == 'VERB': #following lines adapted from subjects_by_verb_count
            doc_verb_list.append(str(word))
            for subject in word.children:
                if (subject.pos_ == 'NOUN' or subject.pos_ =='PRON' or subject.pos_ =='PROPN') and (subject.dep_ =='nsubj' or subject.dep_=='csubj'):
                    doc_nountokenlist.append(str(subject))
                    doc_bigrams_list.append((str(word),str(subject)))
    
    set_subject_list = list(set(doc_nountokenlist))
    set_verb_list = list(set(doc_verb_list))
    set_bigrams_list = list(set(doc_bigrams_list))

    bigramsdict={}
    verbdict={}
    subjectdict={}

    for bigram in set_bigrams_list:
        count=0
        for bigramcheck in doc_bigrams_list:
            if bigram == bigramcheck:
                count+=1
                bigramsdict[bigram]=(count/len(doc_bigrams_list))
    
    for verb in set_verb_list:
        count=0
        for verbcheck in doc_verb_list:
            if verb == verbcheck:
                count+=1
                verbdict[verb]=(count/len(doc_verb_list))

    for subject in set_subject_list:
        count=0
        for subjectcheck in doc_nountokenlist:
            if subject == subjectcheck:
                count+=1
                subjectdict[subject]=(count/len(doc_nountokenlist))
    
    pmi_dict={}
    for i in set_bigrams_list:
        numerator=bigramsdict[i]
        denominator1=verbdict.get(i[0])
        denominator2=subjectdict.get(i[1])
        pmivalue = math.log2((numerator)/(denominator1*denominator2))
        pmi_dict[i]=pmivalue

    pmi_dict=dict(sorted(pmi_dict.items(), key=lambda item:item[1], reverse=True))
    toptenpmi = dict(itertools.islice(pmi_dict.items(),10))
    listkeystop10 = list(toptenpmi.keys())

    return(listkeystop10)




def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    dictionary = {}
    doc_nountokenlist=[]
    for word in doc:
        if word.lemma_ == verb and word.pos_ == 'VERB': #following lines adapted from https://spacy.io/usage/linguistic-features
            for subject in word.children:
                if (subject.pos_ == 'NOUN' or subject.pos_ =='PRON' or subject.pos_ =='PROPN') and (subject.dep_ =='nsubj' or subject.dep_=='csubj'):
                    doc_nountokenlist.append(subject)

    doc_str_list =[str(token) for token in doc_nountokenlist]
    set_list=list(set(doc_str_list))
    for word in set_list:
        count = 0
        for wordcheck in doc_str_list:
            if wordcheck == word:
                count +=1
        dictionary[word] = count
    
    dictionary = dict(sorted(dictionary.items(), key=lambda item:item[1], reverse=True))
    toptensubs = dict(itertools.islice(dictionary.items(),10))
    listkeystop10 = list(toptensubs.keys())
    
    return(listkeystop10)
    pass



def subject_counts(doc):
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    #most hashed out lines of code can be removed as an appropriate for code was implemented in main
    #have left hashed out code in incase addition of this for loop not required
    #if unhashed lines from dictionary to listkeytop10 need to be indented again

    # total_doc_noun_list=[]
    # all_noun_sets=[]
    # dictionary_list =[]
    # titles = df['title']
    # for doc in df['parsed']:
    dictionary ={}
    doc_nountoken_list=[]
    for word in doc:
        if (word.pos_ == 'NOUN' or word.pos_ =='PRON' or word.pos_ =='PROPN') and (word.dep_ =='nsubj' or word.dep_=='csubj'):
            doc_nountoken_list.append(word)
    doc_str_list =[str(token) for token in doc_nountoken_list] #using token as currently each word in likst is token type
    # total_doc_noun_list.append(doc_str_list)
    set_list=list(set(doc_str_list))
    # all_noun_sets.append(set_list)
    
    for word in set_list:
        count = 0
        for wordcheck in doc_str_list:
            if wordcheck == word:
                count +=1
        dictionary[word] = count
    
    dictionary = dict(sorted(dictionary.items(), key=lambda item:item[1], reverse=True))
    toptensubs = dict(itertools.islice(dictionary.items(),10))
    listkeystop10 = list(toptensubs.keys())
        # dictionary_list.append(listkeystop10)
        
    #x=zip(titles,dictionary_list)

    return(listkeystop10)

    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df)
    #print(df.head())
    nltk.download("cmudict")
    d=cmudict.dict()
    print("Type Token Ratios:")
    print(get_ttrs(df))
    print("FKS")
    print(get_fks(df))
    parse(df)
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    #print(subject_counts(df))
    
    for i, row in df.iterrows():
        print("Top Ten Subjects of:")
        print(row["title"])
        print(subject_counts(row["parsed"]))
        print("\n")
       
    for i, row in df.iterrows():
        print("Subject by verb: to say, ordered by frequency for:")
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "say"))
        print("\n")
    
    for i, row in df.iterrows():
        print("Pointwise Mutual Information for:")
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "say"))
        print("\n")
    

