"""
    Anit Mangal
    21CS10005
    Assignment 3 Task A
"""

from glpk import glpk, GLPK
import sys
import re
from tqdm import tqdm
import math
import nltk

K = 200     # Budget

def solveILP(n, rel, red, l):
    """Function to solve the ILP problem

    Args:
        n (int): Number of sentences
        rel (list(n,)): List of relevance scores
        red (list(n, list(n,))): 2D list of redundancy scores
        l (list(n,)): List of sentence lengths

    Returns:
        list(n,): List of binary values indicating whether the sentence is selected or not
    """
    
    # Objective Coefficients
    obj = rel       # ai.rel[i]
    for i in range(n): 
        red[i] = [-num for num in red[i]]   # -aij.red[i][j]
        obj += red[i]
    A = []                      # Coefficients of constraints
    b = []                      # RHS of constraints
    
    # sum(ai.l[i]) <= K
    temp = l + [0 for i in range(n*n)]
    A.append(temp)
    b.append(K)
    
    # aij - ai <= 0
    for i in range(n):
        for j in range(i+1,n):
            temp = [0 for x in range(n + n*n)]
            temp[i] = -1
            temp[n + i*n + j] = 1
            A.append(temp)
            b.append(0)
            
    # aij - aj <= 0
    for i in range(n):
        for j in range(i+1,n):
            temp = [0 for x in range(n + n*n)]
            temp[j] = -1
            temp[n + i*n + j] = 1
            A.append(temp)
            b.append(0)
            
    # ai + aj - aij <= 1
    for i in range(n):
        for j in range(i+1,n):
            temp = [0 for x in range(n + n*n)]
            temp[i] = 1
            temp[j] = 1
            temp[n + i*n + j] = -1
            A.append(temp)
            b.append(1)
    
    # Equality Constraints, not needed here
    A_eq = [[0 for i in range(n + n*n)]]
    b_eq = [0]
    
    # Integer Constraints (All variables are binary)
    intcon = [i for i in range(n + n*n)]
    
    """Solve the ILP Problem:
        c: objective coefficients
        A_ub: Coefficients of the inequality constraints
        b_ub: RHS of the inequality constraints
        A_eq: Coefficients of the equality constraints
        b_eq: RHS of the equality constraints
        bounds: Bounds on the variables (0 <= x <= 1)
        solver: Solver to use (mip for mixed integer programming)
        sense: Maximization or Minimization
        message_level: Verbosity of the solver
        disp: Display the output
        mip_options: Options for the solver
    """
    result = glpk(c = obj, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds = (0,1), solver= 'mip', sense = GLPK.GLP_MAX, message_level=GLPK.GLP_MSG_OFF, disp= False, mip_options={'intcon': intcon})
    
    return [int(result.get('x')[i]) for i in range(n)]

def buildL(sentences):
    """Function to build the list of lengths of sentences

    Args:
        sentences (list(n,)): List of sentences

    Returns:
        list(n,): List of lengths of sentences
    """
    # word_tokenize returns the list of words in the sentence
    l = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    return l

def cosineSimilarity(v1, v2):
    """Function to calculate the cosine similarity between two vectors

    Args:
        v1 (list(float,)): _description_
        v2 (list(float,)): _description_

    Returns:
        float: Cosine similarity between the two vectors
    """
    dotProduct = sum([v1[i]*v2[i] for i in range(len(v1))])
    magnitude1 = math.sqrt(sum([v1[i]*v1[i] for i in range(len(v1))]))
    magnitude2 = math.sqrt(sum([v2[i]*v2[i] for i in range(len(v2))]))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dotProduct/(magnitude1*magnitude2)

def getSentences(text):
    """Function to get the list of sentences from the text

    Args:
        text (string): Text to be tokenized into sentences

    Returns:
        list(string,): List of sentences
    """
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences if len(sentence) > 0]
    return sentences

def buildTextVector(text, V, N, invertedIndex):
    """Function to build the text vector for the given text

    Args:
        text (string): Text to be tokenized
        V (list(string,)): List of words in the vocabulary - vectorized
        N (int): Number of documents
        invertedIndex (dict(string, list(2, list(2,))): Inverted Index

    Returns:
        list(float,): Text Vector
    """
    vector = []
    tokens = preprocessText(text)
    for word in V:
        tf = 0
        if tokens.count(word) > 0:
            tf = 1 + math.log10(tokens.count(word))
        idf = math.log10(N/invertedIndex[word][0])
        vector.append(tf*idf)
    return vector

def buildRel(V, invertedIndex, collectionVector, sentences):
    """Function to build the relevance scores for the sentences

    Args:
        V (list(string,)): List of words in the vocabulary - vectorized
        invertedIndex (dict(string, list(2, list(2,))): Inverted Index
        collectionVector (list(float,)): Collection Vector
        sentences (list(string,)): List of sentences

    Returns:
        list(float,): List of relevance scores
    """
    n = len(sentences)
    rel = [1.0/(i+1) for i in range(n)]     # Pos(i) = 1/(i+1)
    for i in range(n):
        sim = cosineSimilarity(buildTextVector(sentences[i], V, N, invertedIndex), collectionVector)
        rel[i] += sim   # Add the cosine similarity to the relevance score
    return rel

def buildRed(V, invertedIndex, sentences):
    """Function to build the redundancy scores for the sentences

    Args:
        V (list(string,)): List of words in the vocabulary - vectorized
        invertedIndex (dict(string, list(2, list(2,))): Inverted Index
        sentences (list(string,)): List of sentences

    Returns:
        list(n, list(n,)): 2D list of redundancy scores
    """
    n = len(sentences)
    red = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        # Only for j > i
        for j in range(i+1, n):
            red[i][j] = cosineSimilarity(buildTextVector(sentences[i], V, N, invertedIndex), buildTextVector(sentences[j], V, N, invertedIndex))
    return red

def buildCollectionVector(invertedIndex, V, N):
    """Function to build the collection vector

    Args:
        invertedIndex (dict(string, list(2, list(2,))): Inverted Index
        V (list(string,)): List of words in the vocabulary - vectorized
        N (int): Number of documents

    Returns:
        list(float,): Collection Vector
    """
    collectionVector = []
    for word in V:
        tf = sum([invertedIndex[word][1][i][1] for i in range(invertedIndex[word][0])])
        idf = N/invertedIndex[word][0]
        collectionVector.append((1+math.log10(tf))*math.log10(idf))
    return collectionVector

stopWords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()
def preprocessText(text):
    """Function to preprocess the text

    Args:
        text (string): Text to be preprocessed

    Returns:
        list(string,): List of tokens
    """
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stopWords]
    return tokens

def addToInvertedIndex(docId, text, invertedIndex):
    """Function to add the text to the inverted index

    Args:
        docId (string): Document ID
        text (string): Text to be added
        invertedIndex (dict(string, list(2, list(2,))): Inverted Index

    Returns:
        dict(string, list(2, list(2,)): Updated Inverted Index
    """
    text = preprocessText(text)
    for word in text:
        if word not in invertedIndex:
            invertedIndex[word] = [[docId, 1]]
        else:
            found = False
            for i in range(len(invertedIndex[word])):
                if invertedIndex[word][i][0] == docId:
                    invertedIndex[word][i][1] += 1
                    found = True
                    break
            if not found:
                invertedIndex[word].append([docId, 1])
    return invertedIndex

def getDocIdAndText(text):
    """Function to get the Document ID, Text and Highlight from the input

    Args:
        text (string): Input string

    Returns:
        string, string, string: Document ID, Text, Highlight
    """
    docId = ""
    article = ""
    highlight = ""
    # Read from comma separated: docId,article,highlight
    i = 0
    while text[i] != ',':
        docId += text[i]
        i += 1
    i += 1
    # Find last valid comma
    j = len(text) - 1
    if text[j] == '\n':
        j -= 1
    if text[j] == '"':
        j -= 1
        while text[j] != '"' or text[j-1] != ',':
            j -= 1
        j -= 1
    else:
        while text[j] != ',':
            j -= 1
    # Get article and highlight
    article = text[i:j]
    highlight = text[j+1:]
    article = article.strip(' "\n')
    highlight = highlight.strip(' "\n')
    return docId, article, highlight

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Assignment3_21CS10005_summarizer.py <input_file>")
        sys.exit(1)
    
    with open("Assignment3_21CS10005_summary.txt", 'w') as f:
        f.write("DocID,Summary\n")
        
    input_file = sys.argv[1]
    
    # Count the number of rows in the input file
    numRows = -1
    with open(input_file, 'r', encoding="utf8") as f:
        line = f.readline()
        while line:
            numRows += 1
            line = f.readline()
            
    blackListDocs = set()
    invertedIndex = {}
    V = []
    N = 0
    with open(input_file, 'r', encoding="utf8") as f:
        line = f.readline()
        line = f.readline()
        # Progress Bar
        with tqdm(total=numRows, desc="Building Inverted Index") as pbar:
            while line:
                docId, text, highlight= getDocIdAndText(line)
                highlightList = preprocessText(highlight)
                wordCount = len(highlightList)
                # If the word count is greater than K, add the document to the blacklist
                if wordCount > K:
                    blackListDocs.add(docId)
                    pbar.update(1)
                    line = f.readline()
                    continue
                # Update count and inverted index
                N += 1
                invertedIndex = addToInvertedIndex(docId, text, invertedIndex)
                pbar.update(1)
                line = f.readline()
    print(f'Using {N} documents.')
    
    # Sort the inverted index and build the collection vector
    for word in invertedIndex:
        invertedIndex[word].sort()
        invertedIndex[word] = [len(invertedIndex[word]), invertedIndex[word]]
    V = list(invertedIndex.keys())
    
    collectionVector = buildCollectionVector(invertedIndex, V, N)
    
    numRows -= len(blackListDocs)
    with open(input_file, 'r', encoding="utf8") as f:
        line = f.readline()
        line = f.readline()
        with tqdm(total=numRows, desc="Summarizing") as pbar:
            while line:
                docId, text, _ = getDocIdAndText(line)
                if docId in blackListDocs:
                    line = f.readline()
                    continue
                # Get the sentences, relevance, redundancy and length
                sentences = getSentences(text)
                rel = buildRel(V, invertedIndex, collectionVector, sentences)
                red = buildRed(V, invertedIndex, sentences)
                l = buildL(sentences)
                # Solve the ILP problem
                result = solveILP(len(sentences), rel, red, l)
                # Write the summary to the file
                with open("Assignment3_21CS10005_summary.txt", 'a', encoding="utf8") as g:
                    g.write(f"{docId},")
                    for i in range(len(result)):
                        if result[i] == 1:
                            g.write(f"{sentences[i]} ")
                    g.write("\n")
                pbar.update(1)
                line = f.readline()
    print("Summary written to Assignment3_21CS10005_summary.txt")