import numpy as np
from numba import njit, prange


def train(doc_word_mat, n_topics, max_iter=100):
    """
    wrapper function for training a corpus using pLSA. To save computation time, training
    uses maximum number of iterations as the stopping criterion.
    Args:
        doc_word_mat: a document word matrix stored as a sparse matrix
        n_topics: number of topics to be trained
        max_iter: Maximum number of iteration before the EM algorithm stops
    Returns:
        likelihood function value when the training stops
        word_given_topic matrix
        topic_given_word matrix
    """
    doc = doc_word_mat.row
    word = doc_word_mat.col
    freq = doc_word_mat.data

    # Random initialization
    word_given_topic = np.random.rand(doc_word_mat.shape[1], n_topics)
    topic_given_doc = np.random.rand(n_topics, doc_word_mat.shape[0])

    return plsa_sparse(doc, word, freq, word_given_topic, topic_given_doc, max_iter)


@njit(parallel=True)
def plsa_sparse(doc, word, freq, word_given_topic, topic_given_doc, max_iter):
    """
    runs the EM algorithm for pLSA.
    Args:
        doc: row indices of the document word matrix
        word: col indices of the document word matrix
        freq: entries of the document word matrix
        word_given_topic: initialized word_given_topic matrix
        topic_given_doc: initialized topic_given_doc matrix
        max_iter: the maximum number of iterations before the EM algorithm stops
    Returns:
        word_given_topic matrix
        topic_given_word matrix
    """
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
