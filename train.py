import pLSA
import k_means
import lda
import numpy as np
import heapq
import operator
import os
import time
import pickle
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def train(model_name, n_topics, n_runs, doc_word_matrix, model_file, topic_doc_file):
    if n_runs == 0 or (model_name not in ["pLSA", "k_means", "lda"]):
        return

    print("Training " + model_name + f" with {n_topics} topics...")
    best_objective = float("-inf")
    run_stats = []
    for i in range(1, n_runs + 1):
        print(f"Run #{i}... ", end=' ')
        start = time.perf_counter()

        if model_name == "pLSA":
            objective, model, topic_given_doc = pLSA.train(doc_word_matrix, n_topics)
        elif model_name == "k_means":
            objective, model, topic_given_doc = k_means.train(doc_word_matrix, n_topics, eps=1e-9)
        else:
            objective, model, topic_given_doc = lda.train(doc_word_matrix, n_topics, alpha=2, beta=0.01, n_iter=250)

        runtime = time.perf_counter() - start
        run_stats.append([n_topics, runtime, objective])
        print('done. %.5f s' % runtime)
        if objective > best_objective:
            best_objective = objective
            best_model = model
            best_topic_given_doc = topic_given_doc
    print('')

    # Save topics
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    with open(topic_doc_file, 'wb') as f:
        pickle.dump(best_topic_given_doc, f)

    return run_stats


def save_common_words(model_name, n, model_file, topic_doc_file, doc_sizes, index, output_file):
    with open(model_file, 'rb') as f:
        word_given_topic = pickle.loads(f.read())
        if model_name == "k_means":
            word_given_topic = word_given_topic.T

    n_topics = word_given_topic.shape[1]

    with open(topic_doc_file, 'rb') as f:
        topic_doc = pickle.loads(f.read())

    # Compute importance of each topic
    if model_name == "k_means":
        topic_importance = np.zeros((n_topics))
        for t, f in zip(topic_doc, doc_sizes):
            topic_importance[t] += f
        topic_importance /= doc_sizes.sum()
    else:
        topic_importance = (topic_doc * doc_sizes).sum(axis=1) / doc_sizes.sum()

    topic_rank = np.argsort(-topic_importance)

    # Find most common words in each topic
    word_in_topics = []

    for i in range(n_topics):
        j = topic_rank[i]
        word_in_topic = ["{0:.3f}".format(topic_importance[j] * 100)]
        index_list = [index_count[0] for index_count in heapq.nlargest(n,
                                                                       enumerate(word_given_topic[:, j]),
                                                                       key=operator.itemgetter(1))]

        for element in index_list:
            word_in_topic.append(index[element])

        word_in_topics.append(word_in_topic)

    # Save the results
    with open(output_file, 'w') as f:
        for line in word_in_topics:
            f.write(' '.join(line) + '\n')


def main(n_topics=[5],
         n_common_words=20,
         plsa_runs=0,
         k_means_runs=0,
         lda_runs=0,
         corpus_names=[]):

    base_dir = os.getcwd()
    vocab_dir = 'vocabulary_files'
    model_dir = 'models'
    top_words_dir = 'top_words'
    stats_dir = 'stats'

    for corpus_name in corpus_names:
        index_to_word_file = os.path.join(base_dir, vocab_dir, corpus_name, corpus_name + '_iw.txt')
        doc_word_file = os.path.join(base_dir, vocab_dir, corpus_name, corpus_name + '_matrix.npz')

        with open(index_to_word_file, 'rb') as f:
            index = pickle.loads(f.read())

        with open(doc_word_file, 'rb') as f:
            doc_word_matrix = sparse.load_npz(f)

        d, w = doc_word_matrix.shape
        nnz = doc_word_matrix.nnz
        n = np.sum(doc_word_matrix.data)
        print("\nCorpus: " + corpus_name)
        print("Number of documents: %d" % d)
        print("Number of words: %d" % w)
        print("NNZ = {}".format(nnz))
        print("N = {}".format(n))
        print("N / NNZ = {0:.2f}".format(n / nnz))
        print("Sparsity: %.2f %%" % (100 * nnz / (d * w)))
        print('')

        for model_name, n_runs in [("pLSA", plsa_runs), ("k_means", k_means_runs), ("lda", lda_runs)]:
            if n_runs > 0:
                model_file = [os.path.join(base_dir, model_dir,
                                           corpus_name + '_' + model_name + '_' + str(t)
                                           + '_topics.txt') for t in n_topics]
                topic_doc_file = [os.path.join(base_dir, model_dir,
                                           corpus_name + '_' + model_name + '_' + str(t)
                                           + '_topics_doc.txt') for t in n_topics]
                top_words_file = [os.path.join(base_dir, top_words_dir,
                                               corpus_name + '_' + model_name + '_' + str(t)
                                               + '_topics_' + str(n_common_words) + '_words.txt') for t in n_topics]
                stats = []
                for i, t in enumerate(n_topics):
                    stats.extend(train(model_name, t, n_runs, doc_word_matrix, model_file[i], topic_doc_file[i]))
                # Plot times
                stats = np.array(stats)
                plt.figure()
                plt.title(corpus_name + ' - ' + model_name)
                plt.xlabel('# Topics')
                plt.xticks(n_topics)
                plt.ylabel('Time (s)')
                plt.scatter(stats[:, 0], stats[:, 1])
                plt.savefig(os.path.join(base_dir, stats_dir, corpus_name + '_' + model_name + '_times.png'))
                # Plot objectives
                plt.figure()
                plt.title(corpus_name + ' - ' + model_name)
                plt.xlabel('# Topics')
                plt.xticks(n_topics)
                plt.ylabel('Objective')
                plt.scatter(stats[:, 0], stats[:, 2])
                plt.savefig(os.path.join(base_dir, stats_dir, corpus_name + '_' + model_name + '_objectives.png'))
                # Save stats
                np.savetxt(os.path.join(base_dir, stats_dir, corpus_name + '_' + model_name + '_stats.txt'), stats)
                # Save top words
                doc_sizes = doc_word_matrix.toarray().sum(axis=1)
                for i, t in enumerate(n_topics):
                    save_common_words(model_name, n_common_words, model_file[i], topic_doc_file[i], doc_sizes,
                                      index, top_words_file[i])


if __name__ == "__main__":
    main(corpus_names=['National_Gazette', 'Gazette_of_US'],
         n_topics=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
         n_common_words=30,
         plsa_runs=0,
         k_means_runs=0,
         lda_runs=0)
