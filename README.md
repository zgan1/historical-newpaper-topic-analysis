This file gives a short description of what each Python script does.


corpus.py:

Used to preprocess the raw data using spellchecking, stemming, and frequency capping.
It then saves converts a corpus into a document word matrix.
The output is saved in the 'vocabulary_files' folder.


train.py [depends on k_means.py, pLSA.py and lda.py, each file implements the corresponding training algorithm]:

Trains the models and saves the results in the 'models' folder. The values of the objective function and the execution time are plotted and saved in the 'stats' folder.
The most important words in each topic are saved in the 'top_words' folder.
All training algorithms are bag-of-words algorithms.

topic_coherence.py:

Computes the topic coherence score of each topic in each model and plots the resulting distributions.
The topic coherence score assesses the frequency of each pair of word assigned to a topic occuring in the same document.
The output is saved in the 'coherence' folder.


consistency.py [depends on model.py]:

Computes and plots various consistency measures. The results are saved in the 'consistency' and 'self_consistency' folders.
Consistency measures the similarity between topics treating each topic as a unit vector(k-means, normalized using L2 norm) or a probablity mass function
(pLSA, LDA normalized using L1 norm). In the "consistency" folder are the comparisons between topics across two models. 
In the "self_consistency" folder are the comparisons of topics within a model.
