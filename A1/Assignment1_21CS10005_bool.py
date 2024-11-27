import sys
import pickle

def merge(postings):
    result = postings[0]
    for i in range(1, len(postings)):
        result = list(set(result)&set(postings[i]))
    return result

def getResult(query, invertedIndex):
    postings = []
    for word in query:
        if word in invertedIndex:
            postings.append(invertedIndex[word])
        else:
            return []
    if query == 'information science definition possible':
        print(postings)
    return merge(postings)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Invalid number of arguments. Run python Assignment1_21CS10005_bool.py <path to the model> <path to query file>")
        exit(1)
    modelPath = sys.argv[1]
    queryPath = sys.argv[2]
    invertedIndex = pickle.load(open(modelPath, "rb"))
    
    queryText = open(queryPath, "r").readlines()
    with open("Assignment1_21CS10005_results.txt", "w") as file:
        for query in queryText:
            queryId = query.split()[0]
            queryResult = getResult(query.split()[1:-1], invertedIndex)
            
            file.write(f"{queryId}: ")
            if len(queryResult) > 0:
                for documentId in queryResult:
                    file.write(f"{documentId} ")
            file.write("\n")
    print("Results stored in Assignment1_21CS10005_results.txt")