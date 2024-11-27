"""
    Anit Mangal
    21CS10005
    Assignment 1 Task A
"""

import sys
import re
import spacy
import pickle
import os

def getIdAndText(cisiPath):
    """Extracts ID and Text from each document. ID is taken after .I and text is taken after .W

    Args:
        cisiPath (str): Path to CISI directory

    Returns:
        dict: Dictionary containing (document_id, text) pairs.
    """
    documentDict = {}
    with open(cisiPath+"/CISI.ALL", "r") as f:
        regex = re.compile(r'^\.I [0-9]+\n$')
        text = f.readlines()
        for i in range(len(text)):
            # Match the .I line to indicate new document.
            if regex.match(text[i]):
                documentId = int(text[i].split(' ')[1].strip())
                # Get text from .W section
                j = i+1
                while j < len(text) and text[j] != '.W\n':
                    j+=1
                if j == len(text):
                    break
                if (text[j] == '.W\n'):
                    documentDict[documentId] = text[j+1].strip()
                    j+=2
                    while j < len(text) and text[j]!='.X\n':
                        documentDict[documentId] += ' '+ text[j].strip()
                        j+=1
                i = j
    return documentDict

def cleanupText(documentDict):
    """Removes stop words, punctuations and lemmatizes the text

    Args:
        documentDict (dict): Dictionary containing (document_id, text) pairs.

    Returns:
        dict: Dictionary containing (document_id, list_of_tokens) pairs.
    """
    nlp = spacy.load("en_core_web_sm")
    for key in documentDict:
        documentDict[key] = re.sub(r'[^\w\s]', ' ', documentDict[key])   # Remove non-words or non-whitespaces
        documentDict[key] = nlp(documentDict[key].lower())  # Tokenize
        documentDict[key] = [token.lemma_ for token in documentDict[key] if not token.is_punct and not token.is_stop] # Remove punctuation and stop words and lemmatize
        documentDict[key] = [word for word in documentDict[key] if re.match(r'[\w]', word)] # Remove space-only/newline tokens
    return documentDict

def saveIndex(documentDict):
    """Creates and saves an inverted index for tokens

    Args:
        documentDict (dict): Dictionary containing (document_id, list_of_tokens) pairs.
    """
    invertedIndex = {}
    
    for key in documentDict:
        for word in documentDict[key]:
            if word not in invertedIndex:
                invertedIndex[word] = [key]
            else:
                if key not in invertedIndex[word]:
                    invertedIndex[word].append(key)
    
    # Sort to create postings list
    for word in invertedIndex:
        invertedIndex[word].sort()
    
    pickle.dump(invertedIndex, open(f"{os.path.dirname(__file__)}/model_queries_21CS10005.bin", "wb"))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Invalid number of arguments. Run python Assignment1_21CS10005_indexer.py <path to the CISI folder>")
        exit(1)
    cisiPath = sys.argv[1]
    documentsDict = getIdAndText(cisiPath)
    documentsDict = cleanupText(documentsDict)
    saveIndex(documentsDict)
    print("Inverted Index saved in model_queries_21CS10005.bin")
    