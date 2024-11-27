21CS10005

Special libraries required: spacy

Python version: Python 3.10.9

Design:
The design follows the instructions written in the assignment. The spacy library is used to perform lemmatization on the documents and queries. The merging routine uses set intersection to get relevant query results. The inverted index created by using spacy and cleaning up has 7321 words, each having its posting list.

Troubleshooting:
Make sure the path for Task A is the folder containing "CISI.ALL". For Task B, it should be the path to the exact query file (e.g. CISI.QRY). For Task C, the full path to "model_queries_21CS10005.bin" and "queries_21CS10005.txt" should be given.