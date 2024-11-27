"""
    Anit Mangal
    21CS10005
    Assignment 1 Task B
"""
import sys
import re
import spacy
import os

def getIdAndText(queryPath):
    """Extracts ID and Text from each query. ID is taken after .I and text is taken after .W

    Args:
        queryPath (str): Path to query file

    Returns:
        dict: Dictionary containing (query_id, text) pairs.
    """
    queryDict = {}
    with open(queryPath, "r") as f:
        regex = re.compile(r'^\.I [0-9]+\n$')
        text = f.readlines()
        for i in range(len(text)):
            # Match the .I line to indicate new query.
            if regex.match(text[i]):
                # Check if .A or .B section is present
                j = i+1
                valid = True
                while j < len(text) and not regex.match(text[j]):
                    if text[j] == '.A\n' or text[j] == '.B\n':
                        valid = False
                        break
                    j+=1
                if not valid:
                    i = j
                    continue
                queryId = int(text[i].split(' ')[1].strip())
                # Get text from .W section
                j = i+1
                while j < len(text) and text[j] != '.W\n':
                    j+=1
                if j == len(text):
                    break
                if (text[j] == '.W\n'):
                    queryDict[queryId] = text[j+1].strip()
                    j+=2
                    while j < len(text) and not regex.match(text[j]):
                        queryDict[queryId] += ' '+ text[j].strip()
                        j+=1
                i = j
    return queryDict

def cleanupText(queryDict):
    """Removes stop words, punctuations and lemmatizes the text

    Args:
        queryDict (dict): Dictionary containing (query_id, text) pairs.

    Returns:
        dict: Dictionary containing (query_id, list_of_tokens) pairs.
    """
    nlp = spacy.load("en_core_web_sm")
    for key in queryDict:
        queryDict[key] = re.sub(r'[^\w\s]', ' ', queryDict[key])   # Remove non-words or non-whitespaces
        queryDict[key] = nlp(queryDict[key].lower())  # Tokenize
        queryDict[key] = [token.lemma_ for token in queryDict[key] if not token.is_punct and not token.is_stop] # Remove punctuation and stop words and lemmatize
        queryDict[key] = [word for word in queryDict[key] if re.match(r'[\w]', word)] # Remove space-only/newline tokens
    return queryDict

def saveQueries(queryDict):
    """Saves queries in <query id> [TAB] <query text> format

    Args:
        queryDict (dict): Dictionary containing (query_id, list_of_tokens) pairs.
    """
    with open(f"{os.path.dirname(__file__)}/queries_21CS10005.txt", "w") as file:
        for key in queryDict:
            file.write(f"{key}\t{' '.join(queryDict[key])}\n")
            
if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Invalid number of arguments. Run python Assignment1_21CS10005_parser.py <path to the query file>")
        exit(1)
    queryPath = sys.argv[1]
    queryDict = getIdAndText(queryPath)
    queryDict = cleanupText(queryDict)
    saveQueries(queryDict)
    print("Saved queries in queries_21CS10005.txt")