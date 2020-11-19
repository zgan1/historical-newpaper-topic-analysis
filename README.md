This file gives a short description of what each Python script does.


corpus.py:

Used to preprocess the raw data. The output is saved in the 'vocabulary_files' folder.


train.py [depends on k_means.py, pLSA.py and lda.py]:

Trains the models and saves the results in the 'models' folder. The values of the objective function and the execution time are plotted and saved in the 'stats' folder.
The most important words in each topic are saved in the 'top_words' folder.


topic_coherence.py:

Computes the topic coherence score of each topic in each model and plots the resulting distributions. The output is saved in the 'coherence' folder.


consistency.py [depends on model.py]:

Computes and plots various consistency measures. The results are saved in the 'consistency' and 'self_consistency' folders.