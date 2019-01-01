import math
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from nltk import word_tokenize
import nltk
nltk.download('punkt')
from nltk.stem import SnowballStemmer
import pandas as pd

#Initialize Global variables
docIDFDict = {}
avgDocLength = 0
stemmer = SnowballStemmer('english')
totalWordList = []
words = []


def GetCorpus(inputfile,corpusfile, targetfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    fx = open(targetfile, "w", encoding = "utf-8")
    for line in f:
        if line.strip().split("\t")[3] == "1":
            passage = line.strip().lower().split("\t")[2]
            fw.write(passage+"\n")
        fx.write(str(line.strip().split("\t")[3])+"\n")
    f.close()
    fw.close()
    fx.close()

def getTarget(inputfile,targetfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(targetfile,"w",encoding="utf-8")
    for line in f:
        passage = line.strip().lower().split("\t")[3]
        fw.write(passage+"\n")
    f.close()
    fw.close()

def tokenize(text):
    tokens = word_tokenize(text)
    stems = []
    for item in tokens: stems.append(stemmer.stem(item))
    return stems

def TfIdfAlgo(corpusfile, delimiter = ' '):
    
    print("Algo started")
    global totalWordList
    
    for line in open(corpusfile, "r", encoding="utf-8"):
        totalWordList.append(line)
    
    totalWordList = [" ".join(tokenize(txt.lower())) for txt in totalWordList]

    joblib.dump(totalWordList, 'totalWordList.pkl')
    print("Tokenized")

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(totalWordList)
    joblib.dump(matrix, 'matrix.pkl')

    matrix = pd.DataFrame(matrix, columns = vectorizer.get_feature_names())
    top_words = matrix.sum(axis=0).sort_values(ascending=False)

    print(top_words)


def helper(corpusfile, delimiter = ' '):
    
    global words
    
    for line in open(corpusfile, "r", encoding="utf-8"):
        words.append([word for word in line.strip().lower().split(' ')])

# The following IDF_Generator method reads all the passages(docs) and creates Inverse Document Frequency(IDF) scores for each unique word using below formula
# IDF(q_i) = log((N-n(q_i)+0.5)/(n(q_i)+0.5)) where N is the total number of documents in the collection and n(q_i) is the number of documents containing q_i
# After finding IDF scores for all the words, The IDF dictionary will be saved in "docIDFDict.pickle" file in the current directory

def IDF_Generator(corpusfile, delimiter=' ', base=math.e) :
    
    global docIDFDict,avgDocLength
    
    docFrequencyDict = {}
    numOfDocuments = 0
    totalDocLength = 0
    
    for line in open(corpusfile,"r",encoding="utf-8") :
        doc = line.strip().split(delimiter)
        totalDocLength += len(doc)
        
        doc = list(set(doc)) # Take all unique words
        
        for word in doc : #Updates n(q_i) values for all the words(q_i)
            if word not in docFrequencyDict :
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1
        
        numOfDocuments = numOfDocuments + 1
        if (numOfDocuments%5000==0):
            print(numOfDocuments)

    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        docIDFDict[word] = math.log((numOfDocuments - docFrequencyDict[word] + 0.5) / (docFrequencyDict[word] + 0.5), base)
    
    avgDocLength = totalDocLength / numOfDocuments
    print(docIDFDict)
    
#   pickle_out = open("IDFDict.pkl","w") # Saves IDF scores in pickle file, which is optional
    joblib.dump(docIDFDict, "IDFDict.pkl")
#   pickle_out.close()
    
    
    print("NumOfDocuments : ", numOfDocuments)
    print("AvgDocLength : ", avgDocLength)


#The following GetBM25Score method will take Query and passage as input and outputs their similarity score based on the term frequency(TF) and IDF values.
def GetBM25Score(Query, Passage, k1=1.5, b=0.75, delimiter=' ') :
    
    global docIDFDict,avgDocLength
    
    query_words= Query.strip().lower().split(delimiter)
    passage_words = Passage.strip().lower().split(delimiter)
    passageLen = len(passage_words)
    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score

#The following line reads each line from testfile and extracts query, passage and calculates BM25 similarity scores and writes the output in outputfile
def RunBM25OnEvaluationSet(testfile,outputfile):
    
    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        tokens = line.strip().lower().split("\t")
        Query = tokens[1]
        Passage = tokens[2]
        score = GetBM25Score(Query,Passage)
        tempscores.append(score)
        lno+=1
        if(lno%10==0):
            tempscores = [str(s) for s in tempscores]
            scoreString = "\t".join(tempscores)
            qid = tokens[0]
            fw.write(qid+"\t"+scoreString+"\n")
            tempscores=[]
        if(lno%5000==0):
            print(lno)
    print(lno)
    f.close()
    fw.close()

def getRank(inputfile, outputfile):
    f = open(inputfile, 'r')
    fw = open(outputfile, 'w')
    lines = f.readlines()

    counter = 0
    for i in range(len(lines)):
        if lines[i] == "1\n":
            fw.write(str(counter)+"\n")
        counter+=1
        if counter == 9:
            counter = 0

    f.close()
    fw.close()

def tfidf():
    
    print("Algo Started")
    f = open("data.tsv", "r")
    lines = f.readlines()
    
    # Make dictionary of all different words with quality factor -> High if available in correct passage     + Finding document length
    
    print("Making dictionary of words")
    
    noOfDocs = 0
    qualityList = []
    wordsList = []
    docLength = []
    for line in lines:
        noOfDocs += 1
        passage = line.strip().lower().split("\t")[2]
        words = [word for word in passage.split(' ')]
        docLength.append(len(words))
        setWords = set(words)
        words = list(setWords)
        if line.strip().split("\t")[3] == "1":
            for word in words:
                if word not in wordsList:
                    wordsList.append(word)
                    qualityList.append(2)
        else:
            for word in words:
                if word not in wordsList:
                    wordsList.append(word)
                    qualityList.append(1)

    joblib.dump(wordsList, "wordsList.pkl")
    joblib.dump(qualityList, "qualityList.pkl")
    joblib.dump(docLength, "docLength.pkl")
    print("Finished making dictionary of words")
    print("No of doc: ", noOfDocs)
                        
    # Finding Term frequency
    print("Started finding term frequency")
    termFreq = []

    for wordex in wordsList:
        ls = []
        for line in lines:
            counter = 0
            passage = line.strip().lower().split("\t")[2]
            for word in passage.split(' '):
                if word == wordex:
                    counter+=1
            ls.append(counter)
        termFreq.append(ls)

    joblib.dump(termFreq, "termFreq.pkl")
    print("Finished finding term frequency")


    # Finding Inverse Doc Frequeency
    print("Started finding Inverse doc frequency")
    inverseFreq = []

    for wordex in wordsList:
        counter = 0
        for line in lines:
            passage = line.strip().lower().split("\t")[2]
            for word in passage.split(' '):
                if word == wordex:
                    counter+=1
                    break
        inverseFreq.append(counter)

    joblib.dump(inverseFreq, "inverseFreq.pkl")
    print("Started finding Inverse doc frequency")


    # Finding TF-IDF of all words

    # Finding IDF
    idf = []
    for i in range(len(docLength)):
        ans = noOfDocs / inverseFreq[i]
        idf.append(math.log(ans) + 0.5)

    # Finding TFIDF
    print("Started main algorithm...")
    tfidf = []
    for i in range(len(wordsList)):
        ls = termFreq[i]
        ans = 0
        for j in range(len(ls)):
            ans += ((ls[j] / docLength[j]) * idf[j])
        ans = ans / len(ls)
        tfidf.append(ans + qualityList[i])
    print("Finsished main algorithm...")

    # Making dictionary
    dict = {}
    for i in range(len(tfidf)):
        dict[wordsList[i]] = tfidf[i]

    joblib.dump(dict, 'wordDict.pkl')
    print("Dictionary made")


if __name__ == '__main__' :
    
    #    inputFileName = "Data.tsv"   # This file should be in the following format : queryid \t query \t passage \t label \t passageid
    #    testFileName = "eval1_unlabelled.tsv"  # This file should be in the following format : queryid \t query \t passage \t passageid # order of the query
    #corpusFileName = "corpus.tsv"
    #    outputFileName = "answer.tsv"
    #
    #    GetCorpus(inputFileName,corpusFileName)    # Gets all the passages(docs) and stores in corpusFile. you can comment this line if corpus file is already generated
    #    print("Corpus File is created.")
    #IDF_Generator(corpusFileName)   # Calculates IDF scores.
#    #    #RunBM25OnTestData(testFileName,outputFileName)
    #    print("IDF Dictionary Generated.")
    #    RunBM25OnEvaluationSet(testFileName,outputFileName)
    #    print("Submission file created. ")
    
    #   TfIdfAlgo("corpus.tsv")

################################################
#    total = joblib.load('totalWordList.pkl')
#    matrix = joblib.load('matrix.pkl')
#    print(matrix.shape)
#
#    dict = {}
#
#    print(matrix.shape)
#    for i in range(1238605):
#        dict[total[i]] = matrix[i][1]
#
#    joblib.load(dict,'dict.pkl')

#getTarget('data.tsv','target.tsv')

#    getRank('target.tsv', 'rank.tsv')

#GetCorpus('data.tsv','sampleCorpus.tsv', 'sampleTarget.tsv')

################################################

#    list = []
#    average = []
#    count = []
#    total = joblib.load('totalWordList.pkl')
#    matrix = joblib.load('matrix.pkl')
#
#    print(len(total[5]))
#    count = 0
#    newMatrix = matrix[5,:].toarray()
#    for i in newMatrix:
#        for j in i:
#            if j!=0.0:
#                count+=1
#    print(count)
#
#    counter = 0
#    nM = matrix[:,5].toarray()
#    for i in nM:
#        if i[0] != 0.0:
#            counter+=1
#
#    print(counter)

    tfidf()
