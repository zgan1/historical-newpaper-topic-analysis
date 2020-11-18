import numpy as np
from numba import njit, prange


def train(doc_word_mat, n_topics, eps=1e-5, max_iter=100):
    doc = doc_word_mat.row
    word = doc_word_mat.col
    freq = doc_word_mat.data

    # Random initialization
    word_given_topic = np.random.rand(doc_word_mat.shape[1], n_topics)
    topic_given_doc = np.random.rand(n_topics, doc_word_mat.shape[0])

    return plsa_sparse(doc, word, freq, word_given_topic, topic_given_doc, eps, max_iter)


@njit(parallel=True)
def plsa_sparse(doc, word, freq, word_given_topic, topic_given_doc, eps, max_iter):
    n_topics, n_docs = topic_given_doc.shape
    n_words = word_given_topic.shape[0]
    nnz = freq.size

    # Auxiliary variables used in the loop
    topic_given_doc_word = np.zeros((n_topics, nnz))
    nnz_sum = np.zeros((nnz))
    topic_sum = np.zeros((n_topics))
    doc_sum = np.zeros((n_docs))

    # Compute document probabilities
    doc_prob = np.zeros((n_docs))

    for i in range(nnz):
        doc_prob[doc[i]] += freq[i]
    doc_prob = doc_prob / np.sum(doc_prob)

    iter = 0
    likelihood = 0
    while True:
        # Expectation step
        nnz_sum[:] = 0
        for i in prange(nnz):
            for t in range(n_topics):
                topic_given_doc_word[t, i] = topic_given_doc[t, doc[i]] * word_given_topic[word[i], t]
                nnz_sum[i] += topic_given_doc_word[t, i]

        # Normalize topic_given_doc_word
        for t in range(n_topics):
            for i in range(nnz):
                topic_given_doc_word[t, i] /= 1e-20 + nnz_sum[i]

        # Maximization step
        word_given_topic[:] = 0
        topic_sum[:] = 0
        topic_given_doc[:] = 0
        doc_sum[:] = 0
        for t in prange(n_topics):
            for i in range(nnz):
                update = freq[i] * topic_given_doc_word[t, i]
                word_given_topic[word[i], t] += update
                topic_sum[t] += update
                topic_given_doc[t, doc[i]] += update
                doc_sum[doc[i]] += update

        # Normalize word_given_topic
        for w in range(n_words):
            for t in range(n_topics):
                word_given_topic[w, t] /= 1e-20 + topic_sum[t]

        # Normalize topic_given_doc
        for t in range(n_topics):
            for d in range(n_docs):
                topic_given_doc[t, d] /= 1e-20 + doc_sum[d]

        # # Compute likelihood
        # old_likelihood = likelihood
        #
        # likelihood = 0
        # for i in range(nnz):
        #     update = 0
        #     for t in range(n_topics):
        #         update += word_given_topic[word[i], t] * topic_given_doc[t, doc[i]]
        #     likelihood += freq[i] * update * doc_prob[doc[i]]
        #
        # #Decide if algorithm should stop
        # if (iter > 0) and (abs(likelihood - old_likelihood) < eps or iter == max_iter):
        #     break
        if iter == max_iter:
            break
        iter += 1

    likelihood = 0
    for i in range(nnz):
        update = 0
        for t in range(n_topics):
            update += word_given_topic[word[i], t] * topic_given_doc[t, doc[i]]
        likelihood += freq[i] * update * doc_prob[doc[i]]

    return likelihood, word_given_topic, topic_given_doc
