21CS10005

Special Libraries required: tqdm, spacy
Python version: 3.10.9

Design of Inverted Index:
The format of the index is assumed to be: {t : [DF(t), [[d, TF(t, d)]]]}
So the inverted index is a dictionary of terms, each having a 2-element list of document frequency for that term along with the postings list. The postings list has a 2-element list as its element per document, which contains document ID (cord_uid) and the term frequency of the term in that document.

Optimisation in ranking:
It is not required to cosine normalise the score vectors in the algorithm used. We only normalise the document vectors.

Commands used:
python Assignment2_21CS10005_ranker.py "<absolute_path_to_folder_to_model_queries>\model_queries_21CS10005.bin" "<absolute_path_to_folder_to_topics-rnd5.xml>\topics-rnd5.xml"

python Assignment2_21CS10005_evaluator.py "<absolute_path_to_folder_to_qrels_file>\qrels-covid_d5_j0.5-5.txt" "<absolute_path_to_folder_to_specific_ranked_list>\Assignment2_21CS10005_ranked_list_<K>.txt"