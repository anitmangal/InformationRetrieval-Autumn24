"""
    Anit Mangal
    21CS10005
    Assignment 2 Task B
"""

import sys
import math

def getRelevance(goldStd, docId):
    """Function to get the relevance of a document

    Args:
        goldStd (list(tuple(string, int))): List of tuples containing document id and relevance
        docId (string): Document id

    Returns:
        int: Relevance of the document
    """
    for i in range(len(goldStd)):
        if goldStd[i][0] == docId:
            return goldStd[i][1]
    return 0

def getAveragePrecision(goldStd, rankedList, k):
    """Function to calculate the average precision at k

    Args:
        goldStd (list(tuple(string, int))): List of tuples containing document id and relevance
        rankedList (list(string)): List of document ids in ranked order
        k (int): Number of documents to consider

    Returns:
        float: Average precision at k
    """
    numRelevantDocs = 0
    sumPrecision = 0
    for i in range(min(k, len(rankedList))):
        docId = rankedList[i]
        rel = getRelevance(goldStd, docId)
        # Take the document as relevant even if it is partially relevant
        if rel > 0:
            numRelevantDocs += 1
            sumPrecision += numRelevantDocs / (i + 1)
    if numRelevantDocs == 0:
        return 0
    return sumPrecision / numRelevantDocs

def getNDCG(goldStd, rankedList, k):
    """Function to calculate the Normalized Discounted Cumulative Gain at k

    Args:
        goldStd (list(tuple(string, int))): List of tuples containing document id and relevance
        rankedList (list(string)): List of document ids in ranked order
        k (int): Number of documents to consider

    Returns:
        float: NDCG at k
    """
    # Sort the gold standard list in decreasing order of relevance to get the ideal ranked list
    idealRankedList = sorted(goldStd, key=lambda x: x[1], reverse=True)
    idealDcg = 0
    dcg = 0
    # Calculate the ideal DCG and DCG, dcg = sum(relevance(i) / log2(i + 2)), i is 0-indexed
    for i in range(min(k, len(rankedList))):
        idealDcg += idealRankedList[i][1] / math.log2(i + 2)
        dcg += getRelevance(goldStd, rankedList[i]) / math.log2(i + 2)
    return dcg/idealDcg

def extractGoldStd(goldStdPath):
    """Function to extract the gold standard from the file

    Args:
        goldStdPath (string): Path to the gold standard file

    Returns:
        dict(int, list(tuple(string, int))): Dictionary containing query id as key and list of tuples containing document id and relevance as
    """
    goldStd = {}
    with open(goldStdPath, 'r') as f:
        for line in f:
            line = line.strip().split()
            queryId = int(line[0])
            docId = line[2]
            rel = int(line[3])
            if queryId not in goldStd:
                goldStd[queryId] = []
            goldStd[queryId].append((docId, rel))
    return goldStd

def extractRankedList(rankedListPath):
    """Function to extract the ranked list from the file

    Args:
        rankedListPath (string): Path to the ranked list file

    Returns:
        dict(int, list(string)): Dictionary containing query id as key and list of document ids in ranked order as value
    """
    rankedList = {}
    with open(rankedListPath, 'r') as f:
        for line in f:
            line = line.strip().split()
            queryId = int(line[0][:-1]) # Remove the colon from the query id
            rankedList[queryId] = []
            for i in range(1, min(21, len(line))):      # Ge the top 20 documents
                rankedList[queryId].append(line[i])
    return rankedList

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 Assignment2_21CS10005_evaluator.py <path_to_gold_standard_ranked_list.txt> <path_to_Assignment2_21CS10005_ranked_list_<K>.txt>")
        exit(1)
    goldStdPath = sys.argv[1]
    goldStdDict = extractGoldStd(goldStdPath)
    
    rankedListPath = sys.argv[2]
    rankedListDict = extractRankedList(rankedListPath)
    
    rankedListCode = rankedListPath.split('_')[-1].split('.')[0]
    
    formatRow = "{:<20}" * 5

    with open(f"Assignment2_21CS10005_metrics_{rankedListCode}.txt", 'w') as f:
        map_10 = 0
        map_20 = 0
        averNDCG_10 = 0
        averNDCG_20 = 0
        numQueries = 0
        f.write(formatRow.format("Query Id", "AP@10", "AP@20", "NDCG@10", "NDCG@20") + "\n")
        for queryId in rankedListDict:
            if queryId not in goldStdDict:
                print("Query Id", queryId, "not found in gold standard")
                continue
            numQueries += 1
            
            goldStd = goldStdDict[queryId]
            rankedList = rankedListDict[queryId]
            
            ap_10 = getAveragePrecision(goldStd, rankedList, 10)
            map_10 += ap_10
            
            ap_20 = getAveragePrecision(goldStd, rankedList, 20)
            map_20 += ap_20
            
            ndcg_10 = getNDCG(goldStd, rankedList, 10)
            averNDCG_10 += ndcg_10
            
            ndcg_20 = getNDCG(goldStd, rankedList, 20)
            averNDCG_20 += ndcg_20
            f.write(formatRow.format(str(queryId), round(ap_10, 4), round(ap_20, 4), round(ndcg_10, 4), round(ndcg_20, 4)) + "\n")
        map_10 /= numQueries
        map_20 /= numQueries
        averNDCG_10 /= numQueries
        averNDCG_20 /= numQueries
        f.write("\nMAP@10: " + str(round(map_10, 4)) + "\n")
        f.write("MAP@20: " + str(round(map_20,4)) + "\n")
        f.write("Average NDCG@10: " + str(round(averNDCG_10,4)) + "\n")
        f.write("Average NDCG@20: " + str(round(averNDCG_20,4)) + "\n")
        print("Metrics written to Assignment2_21CS10005_metrics_" + rankedListCode + ".txt")