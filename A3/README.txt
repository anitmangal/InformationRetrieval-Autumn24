21CS10005

Special Libraries required: tqdm, nltk, scikit-glpk
There were issues while installing proprietary glpk, so scikit-glpk was installed instead which encapsulates the glpk solver.
Python version: 3.10.9

Design of Inverted Index:
The format of the index is assumed to be: {t : [DF(t), [[d, TF(t, d)]]]}
So the inverted index is a dictionary of terms, each having a 2-element list of document frequency for that term along with the postings list. The postings list has a 2-element list as its element per document, which contains document ID (cord_uid) and the term frequency of the term in that document.

To calculate the number of words in highlights, tokenization and stemming are performed.

Commands used:
python Assignment3_21CS10005_summarizer.py "<path_to_data_file"

python Assignment3_21CS10005_evaluator.py "<path_to_data_file>" "<path_to_Assignment3_21CS10005_summary.txt>"