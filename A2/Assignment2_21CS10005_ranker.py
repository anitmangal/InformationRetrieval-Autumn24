"""
    Anit Mangal
    21CS10005
    Assignment 2 Task A
"""
import sys
import xml.etree.ElementTree as ET
import pickle
import math
import re
import spacy
from tqdm import tqdm


def getTfLog(V, invertedIndex, docId):
    """Function to calculate the log normalized term frequency of terms in a document

    Args:
        V (list(string)): Vectorized-list of terms in the vocabulary
        invertedIndex ({string: [int, list([string, int])] }): Inverted Index of the corpus
        docId (string): Document ID

    Returns:
        list(float): List of log normalized term frequency of terms in the document (vectorized)
    """
    tfLog = []
    for term in V:
        # Find the position of the document in the posting list of the term
        pos = -1
        for i in range(invertedIndex[term][0]):
            if invertedIndex[term][1][i][0] == docId:
                pos = i
                break
        # If the document is not present in the posting list, the term frequency is 0
        if pos == -1:
            tfLog.append(0)
        else:
            tfLog.append(1 + math.log(invertedIndex[term][1][pos][1]))
    return tfLog

def getTfAugmented(V, invertedIndex, docId):
    """Function to calculate the augmented term frequency of terms in a document

    Args:
        V (list(string)): Vectorized-list of terms in the vocabulary
        invertedIndex ({string: [int, list([string, int])] }): Inverted Index of the corpus
        docId (string): Document ID

    Returns:
        list(float): List of augmented term frequency of terms in the document (vectorized)
    """
    maxTermFreq = 0         # Maximum term frequency in the document
    for term in V:
        pos = -1
        for i in range(invertedIndex[term][0]):
            if invertedIndex[term][1][i][0] == docId:
                pos = i
                break
        if pos != -1:
            maxTermFreq = max(maxTermFreq, invertedIndex[term][1][pos][1])
            
    tfAugmented = []
    for term in V:
        pos = -1
        for i in range(invertedIndex[term][0]):
            if invertedIndex[term][1][i][0] == docId:
                pos = i
                break
        # If the document is not present in the posting list, the term frequency is 0
        if pos == -1:
            tfAugmented.append(0)
        else:
            tfAugmented.append(0.5 + 0.5 * invertedIndex[term][1][pos][1] / maxTermFreq)
    return tfAugmented

def getIdf(V, invertedIndex, N):
    """Function to calculate the Inverse Document Frequency of terms in the vocabulary

    Args:
        V (list(string)): Vectorized-list of terms in the vocabulary
        invertedIndex ({string: [int, list([string, int])] }): Inverted Index of the corpus
        N (int): Total number of documents in the corpus

    Returns:
        list(float): List of Inverse Document Frequency of terms in the vocabulary
    """
    idf = []
    for term in V:
        idf.append(math.log(N/invertedIndex[term][0]))
    return idf

def getProbIdf(V, invertedIndex, N):
    """Function to calculate the probabilistic Inverse Document Frequency of terms in the vocabulary

    Args:
        V (list(string)): Vectorized-list of terms in the vocabulary
        invertedIndex ({string: [int, list([string, int])] }): Inverted Index of the corpus
        N (int): Total number of documents in the corpus
        
    Returns:
        list(float): List of probabilistic Inverse Document Frequency of terms in the vocabulary
    """
    probIdf = []
    for term in V:
        df = invertedIndex[term][0]
        if (N <= 2*df):
            probIdf.append(0)
        else:
            probIdf.append(math.log(N-df/df))
    return probIdf

def cosineNormalize(v):
    """Function to normalize a vector using cosine normalization

    Args:
        v (list(float)): Vector to be normalized

    Returns:
        list(float): Normalized vector
    """
    length = sum([v[i]*v[i] for i in range(len(v))])
    length = length**0.5
    if length == 0:
        return v
    return [v[i]/length for i in range(len(v))]


def lnc_ltc(V, N, invertedIndex, docIds, queryDict):
    """Function to rank the documents based on the queries using the lnc.ltc scheme

    Args:
        V (list(string)): Vectorized-list of terms in the vocabulary
        N (int): Total number of documents in the corpus
        invertedIndex ({string: [int, list([string, int])] }): Inverted Index of the corpus
        docIds (list(string)): List of document IDs
        queryDict ({string: list(string)}): Dictionary of queries

    Returns:
        {string: list([string, float])}: Dictionary of ranked lists of top 50 documents for each query
    """
    docVectorDict = {}
    for docId in tqdm(docIds, desc="Building Document Vectors", total=len(docIds)):
        tfLog = getTfLog(V, invertedIndex, docId)
        docVectorDict[docId] = cosineNormalize(tfLog)
    
    idf = getIdf(V, invertedIndex, N)
    
    queryRankedLists = {}       # Dictionary to store the ranked lists of top 50 documents for each query
    for queryId in queryDict:
        score = {}      # Stores the score of each document for the query
        
        termDict = {}    # Stores the term frequency of each term in the query
        for term in queryDict[queryId]:
            if term not in V:
                continue
            if term not in termDict:
                termDict[term] = 1
            else:
                termDict[term] += 1
                
        if len(termDict) == 0:
            print(f"Query {queryId} has no terms in the vocabulary: ", end="")
            print(queryDict[queryId])
            continue
        
        # Calculate optimised cosine similarity of the query with each document
        for term in termDict:
            termPos = V.index(term)
            termDict[term] = (1 + math.log(termDict[term])) * idf[termPos]          # w(t,q) = (1 + log(tf(t,q))) * idf(t)
            # Iterate over the posting list of the term to add to the score of each document
            for docTfPair in invertedIndex[term][1]:
                docId = docTfPair[0]
                if docId not in score:
                    score[docId] = 0
                score[docId] += termDict[term] * docVectorDict[docId][termPos]         # score[d] += w(t,q) * w(t,d)
        score = sorted(score.items(), key=lambda x: x[1], reverse=True)         # Sort the documents based on the score
        del score[50:]      # Keep only the top 50 documents
        queryRankedLists[queryId] = score
        
    return queryRankedLists



def lnc_Ltc(V, N, invertedIndex, docIds, queryDict):
    """Function to rank the documents based on the queries using the lnc.Ltc scheme

    Args:
        V (list(string)): Vectorized-list of terms in the vocabulary
        N (int): Total number of documents in the corpus
        invertedIndex ({string: [int, list([string, int])] }): Inverted Index of the corpus
        docIds (list(string)): List of document IDs
        queryDict ({string: list(string)}): Dictionary of queries

    Returns:
        {string: list([string, float])}: Dictionary of ranked lists of top 50 documents for each query
    """
    docVectorDict = {}
    for docId in tqdm(docIds, desc="Building Document Vectors", total=len(docIds)):
        tfLog = getTfLog(V, invertedIndex, docId)
        docVectorDict[docId] = cosineNormalize(tfLog)
    
    idf = getIdf(V, invertedIndex, N)
    
    queryRankedLists = {}
    for queryId in queryDict:
        score = {}
        
        termDict = {}
        sumTf = 0       # Total term frequency of the query
        for term in queryDict[queryId]:
            if term not in V:
                continue
            sumTf += 1
            if term not in termDict:
                termDict[term] = 1
            else:
                termDict[term] += 1
                
        if len(termDict) == 0:
            print(f"Query {queryId} has no terms in the vocabulary: ", end="")
            print(queryDict[queryId])
            continue
        
        avgTf = sumTf/len(termDict)     # Average term frequency of the query
        for term in termDict:
            termPos = V.index(term)
            termDict[term] = (1 + math.log(termDict[term])) * idf[termPos] / (1 + math.log(avgTf))
            for docTfPair in invertedIndex[term][1]:
                docId = docTfPair[0]
                if docId not in score:
                    score[docId] = 0
                score[docId] += termDict[term] * docVectorDict[docId][termPos]
        score = sorted(score.items(), key=lambda x: x[1], reverse=True)
        del score[50:]
        queryRankedLists[queryId] = score
        
    return queryRankedLists

def anc_apc(V, N, invertedIndex, docIds, queryDict):
    """Function to rank the documents based on the queries using the anc.apc scheme
    
    Args:
        V (list(string)): Vectorized-list of terms in the vocabulary
        N (int): Total number of documents in the corpus
        invertedIndex ({string: [int, list([string, int])] }): Inverted Index of the corpus
        docIds (list(string)): List of document IDs
        queryDict ({string: list(string)}): Dictionary of queries
        
    Returns:
        {string: list([string, float])}: Dictionary of ranked lists of top 50 documents for each query
    """
    docVectorDict = {}
    for docId in tqdm(docIds, desc="Building Document Vectors", total=len(docIds)):
        tfLog = getTfAugmented(V, invertedIndex, docId)
        docVectorDict[docId] = cosineNormalize(tfLog)
        
    probIdf = getProbIdf(V, invertedIndex, N)
    
    queryRankedLists = {}
    for queryId in queryDict:
        score = {}
        
        termDict = {}
        for term in queryDict[queryId]:
            if term not in V:
                continue
            if term not in termDict:
                termDict[term] = 1
            else:
                termDict[term] += 1
        maxTf = max(termDict.values())    # Maximum term frequency of the query
        
        for term in termDict:
            termPos = V.index(term)
            termDict[term] = (0.5 + 0.5 * termDict[term] / maxTf) * probIdf[termPos]
            for docTfPair in invertedIndex[term][1]:
                docId = docTfPair[0]
                if docId not in score:
                    score[docId] = 0
                score[docId] += termDict[term] * docVectorDict[docId][termPos]
        score = sorted(score.items(), key=lambda x: x[1], reverse=True)
        del score[50:]
        queryRankedLists[queryId] = score
        
    return queryRankedLists
    

def extractQueries(queriesPath):
    """Function to extract the queries from the XML file

    Args:
        queriesPath (string): Path to the XML file containing the queries

    Returns:
        {string: string}: Dictionary of queries
    """
    queryDict = {}
    try:
        tree = ET.parse(queriesPath)
    except:
        print("Error: Unable to parse the XML file")
        exit(1)
    if tree is None:
        print("Error: Unable to parse the XML file")
        exit(1)
    root = tree.getroot()
    for children in root:
        if children.tag == "topic":
            queryId = children.attrib["number"]
            for child in children:
                if child.tag == "query":
                    query = child.text
                    queryDict[queryId] = query
    print(f"Total number of queries: {len(queryDict)}")
    return queryDict

def processQueries(queryDict):
    """Function to process the queries

    Args:
        queryDict ({string: string}): Dictionary of queries

    Returns:
        {string: list(string)}: Dictionary of processed queries
    """
    nlp = spacy.load("en_core_web_sm")
    for queryId in tqdm(queryDict, desc="Processing Queries", total=len(queryDict)):
        query = queryDict[queryId]
        query = nlp(query.lower())
        query = [token.lemma_ for token in query if not token.is_stop and not token.is_punct]
        query = [word for word in query if re.match(r'[\w]', word)]
        queryDict[queryId] = query
    return queryDict

def extractDocIds(invertedIndex):
    """Function to extract the document IDs from the inverted index

    Args:
        invertedIndex ({string: [int, list([string, int])] }): Inverted Index of the corpus

    Returns:
        list(string): List of document IDs
    """
    docIds = []
    for term in invertedIndex:
        for docTfPair in invertedIndex[term][1]:
            if docTfPair[0] not in docIds:
                docIds.append(docTfPair[0])
    print(f"Total number of documents: {len(docIds)}")
    return docIds

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python Assignment2_21CS10005_ranker.py <path_to_model_queries_21CS10005.bin> <path_to_topics-rnd5.xml>")
        exit(1)
    
    # Get inverted index
    indexPath = sys.argv[1]
    try:
        invertedIndex = pickle.load(open(indexPath, "rb"))
    except:
        print("Error: Unable to load the inverted index")
        exit(1)
    if invertedIndex is None:
        print("Error: Unable to load the inverted index")
        exit(1)
    
    # Get queries
    queriesPath = sys.argv[2]
    queryDict = extractQueries(queriesPath)
    queryDict = processQueries(queryDict)
    
    docIds = extractDocIds(invertedIndex)       # Extract the document IDs from the inverted index
    V = list(invertedIndex.keys())      # Vectorized-list of terms in the vocabulary
    N = len(docIds)         # Total number of documents in the corpus
            
    print("\n\nRanking the documents based on the queries")
    
    print("\nUsing lnc.ltc scheme:")
    with open("Assignment2_21CS10005_ranked_list_A.txt", "w") as f:
        queryRankedLists = lnc_ltc(V, N, invertedIndex, docIds, queryDict)
        for queryId in queryDict:
            f.write(f"{queryId}: ")
            for docId, score in queryRankedLists[queryId]:
                f.write(f"{docId} ")
            f.write("\n")
    print("List saved in Assignment2_21CS10005_ranked_list_A.txt successfully.")
    
    print("\nUsing lnc.Ltc scheme:")
    with open("Assignment2_21CS10005_ranked_list_B.txt", "w") as f:
        queryRankedLists = lnc_Ltc(V, N, invertedIndex, docIds, queryDict)
        for queryId in queryDict:
            f.write(f"{queryId}: ")
            for docId, score in queryRankedLists[queryId]:
                f.write(f"{docId} ")
            f.write("\n")
    print("List saved in Assignment2_21CS10005_ranked_list_B.txt successfully.")
    
    print("\nUsing anc.apc scheme:")
    with open("Assignment2_21CS10005_ranked_list_C.txt", "w") as f:
        queryRankedLists = anc_apc(V, N, invertedIndex, docIds, queryDict)
        for queryId in queryDict:
            f.write(f"{queryId}: ")
            for docId, score in queryRankedLists[queryId]:
                f.write(f"{docId} ")
            f.write("\n")
    print("List saved in Assignment2_21CS10005_ranked_list_C.txt successfully.")