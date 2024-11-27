"""
    Anit Mangal
    21CS10005
    Assignment 3 Task B
"""

import sys
import nltk
import re

def rouge1(highlight, summary):
    """Function to calculate Rouge-1 Precision, Recall and F1 score

    Args:
        highlight (list): List of tokens in highlight
        summary (list): List of tokens in summary

    Returns:
        float, float, float: Rouge-1 Precision, Recall and F1 score
    """
    # Create a dictionary of tokens in highlight and their frequency
    highlictDict = {}
    for token in highlight:
        if token in highlictDict:
            highlictDict[token] += 1
        else:
            highlictDict[token] = 1
            
    # Create a dictionary of tokens in summary and their frequency
    summaryDict = {}
    for token in summary:
        if token in summaryDict:
            summaryDict[token] += 1
        else:
            summaryDict[token] = 1
            
    # Calculate the number of common tokens
    common = 0
    for token in summaryDict:
        if token in highlictDict:
            common += min(highlictDict[token], summaryDict[token])
    
    prec = common / len(summary) if len(summary) != 0 else 0
    recall = common / len(highlight) if len(highlight) != 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0
    return prec, recall, f1

def rouge2(highlight, summary):
    """Function to calculate Rouge-2 Precision, Recall and F1 score

    Args:
        highlight (list): List of tokens in highlight
        summary (list): List of tokens in summary

    Returns:
        float, float, float: Rouge-2 Precision, Recall and F1 score
    """
    # Create a dictionary of bigrams in highlight and their frequency
    highlightDict = {}
    for i in range(len(highlight)-1):
        token = highlight[i] + ' ' + highlight[i+1]
        if token in highlightDict:
            highlightDict[token] += 1
        else:
            highlightDict[token] = 1
    # Create a dictionary of bigrams in summary and their frequency
    summaryDict = {}
    for i in range(len(summary)-1):
        token = summary[i] + ' ' + summary[i+1]
        if token in summaryDict:
            summaryDict[token] += 1
        else:
            summaryDict[token] = 1
            
    # Calculate the number of common bigrams
    common = 0
    for token in summaryDict:
        if token in highlightDict:
            common += min(highlightDict[token], summaryDict[token])
            
    prec = common / len(summary) if len(summary) != 0 else 0
    recall = common / len(highlight) if len(highlight) != 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0
    return prec, recall, f1

stemmer = nltk.stem.PorterStemmer()
stopWords = nltk.corpus.stopwords.words('english')
def preprocess(text):
    """Function to preprocess the text

    Args:
        text (str): Input text

    Returns:
        list: List of tokens after preprocessing
    """
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stopWords and not re.match(r'^\s*$', token)]
    return tokens

def getDocIdAndHighlight(line):
    """Function to extract docId and highlight from the line

    Args:
        line (str): Input line

    Returns:
        str, str: docId, highlight
    """
    docId = ""
    highlight = ""
    
    i = 0
    while line[i] != ',':
        docId += line[i]
        i+=1
    
    j = len(line)-1
    if line[j] == '\n':
        j-=1
    if line[j] == '"':
        j-=1
        while line[j] != '"' or line[j-1] != ',':
            j-=1
        j-=1
    else:
        while line[j] != ',':
            j-=1
    highlight = line[j:]
    return docId, highlight

def getDocIdAndSummary(line):
    """Function to extract docId and summary from the line

    Args:
        line (str): Input line

    Returns:
        str, str: docId, summary
    """
    docId = ""
    summary = ""
    # Read from comma separated: docId,summary
    i = 0
    while line[i] != ',':
        docId += line[i]
        i += 1
    i += 1
    summary = line[i:]
    return docId, summary

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python Assignment3_21CS10005_evaluator.py <path_to_data file> <path to Assignment3_21CS10005_summary.txt>")
        sys.exit(1)
    data_file = sys.argv[1]
    summary_file = sys.argv[2]
            
    with open(data_file, 'r', encoding="utf8") as fd:
        with open(summary_file, 'r', encoding="utf8") as fs:
            refLine = fd.readline()
            summLine = fs.readline()
            
            refLine = fd.readline()
            summLine = fs.readline()
            while refLine and summLine:
                refDocId, refHighlight = getDocIdAndHighlight(refLine)
                summDocId, summ = getDocIdAndSummary(summLine)
                # Find the reference highlight for the summary
                while refDocId != summDocId:
                    refLine = fd.readline()
                    refDocId, refHighlight = getDocIdAndHighlight(refLine)
                    
                highlightTokens = preprocess(refHighlight)
                summaryTokens = preprocess(summ)
                
                r1Prec, r1Recall, r1F1 = rouge1(highlightTokens, summaryTokens)
                r2Prec, r2Recall, r2F1 = rouge2(highlightTokens, summaryTokens)                    
                
                print("\nDocId: ", refDocId)
                print("Rouge-1 Precision / Recall / F1: ", r1Prec, '/', r1Recall, '/', r1F1)
                print("Rouge-2 Precision / Recall / F1: ", r2Prec, '/', r2Recall, '/', r2F1)
                    
                refLine = fd.readline()
                summLine = fs.readline()