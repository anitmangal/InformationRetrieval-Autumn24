import tarfile
import csv
import json
import re
import os
import pickle
from tqdm import tqdm
import spacy

DOCUMENT_LIMIT = -1

def buildInvertedIndex():
    invertedIndex = {}
    nlp = spacy.load("en_core_web_sm")
    numRows = 0
    with open('metadata.csv', 'r', encoding='utf-8') as metadataFile:
        metadata = csv.DictReader(metadataFile)
        for row in metadata:
            numRows += 1
    if (DOCUMENT_LIMIT >= 0): numRows = DOCUMENT_LIMIT
    cnt = 0
    with open('metadata.csv', 'r', encoding='utf-8') as metadataFile:
        metadata = csv.DictReader(metadataFile)
        for row in tqdm(metadata, desc = "Building Inverted Index", total=numRows):
            
            # temporary limit to test on a subset
            cnt += 1
            if (DOCUMENT_LIMIT >= 0 and cnt > DOCUMENT_LIMIT): break
            
            cord_id = row['cord_uid']
            text = row['abstract']
            text = re.sub(r'[^\w\s]', ' ', text)
            text = nlp(text.lower())
            text = [token.lemma_ for token in text if not token.is_stop and not token.is_punct]
            text = [word for word in text if re.match(r'[\w]', word)]
            for word in text:
                if word not in invertedIndex:
                    invertedIndex[word] = [[cord_id, 1]]
                else:
                    pos = -1
                    for i in range(len(invertedIndex[word])):
                        if invertedIndex[word][i][0] == cord_id:
                            pos = i
                            break
                    if pos == -1:
                        invertedIndex[word].append([cord_id, 1])
                    else:
                        invertedIndex[word][pos][1] += 1
    for key in invertedIndex:
        invertedIndex[key].sort()
        invertedIndex[key] = [len(invertedIndex[key]), invertedIndex[key]]
    return invertedIndex

if __name__=="__main__":
    
    invertedIndex = buildInvertedIndex()
    
    print("Inverted Index built successfully")
    
    ''' Format of inverted index:
    {t : [DF(t), [[d, TF(t, d)]]]}
    '''
    
    pickle.dump(invertedIndex, open(f"{os.path.dirname(__file__)}/model_queries_21CS10005.bin", "wb"))
    
    print("Inverted Index saved successfully")