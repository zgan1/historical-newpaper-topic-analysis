import numpy as np
import scipy.sparse as sparse
from numba import njit, prange
import scipy.special as sc
import matplotlib.pyplot as plt
import time


def train(doc_word_mat, n_topics, alpha, beta, n_iter, plot_likelihood=False, plot_file=''):
    N = np.sum(doc_word_mat.data)
    n_docs = doc_word_mat.shape[0]
    vocab_size = doc_word_mat.shape[1]
    n_topics = n_topics

    doc = np.repeat(doc_word_mat.row, doc_word_mat.data)
    word = np.repeat(doc_word_mat.col, doc_word_mat.data)
    topic = np.random.randint(0, n_topics, N)
    prob = np.zeros((n_topics))

    topic_doc = np.zeros((n_topics, n_docs))
    word_topic = np.zeros((vocab_size, n_topics))
    topic_count = np.zeros((n_topics))
    initialize(N, topic_doc, word_topic, topic_count, doc, word, topic)

    if plot_likelihood:
        n_points = 400
        step = n_iter // n_points
        iterations = [step * (i + 1) for i in range(n_points)]
        likelihoods = []
        for i in range(n_points):
            gibbs_sampling(N, vocab_size, n_topics, doc, word, topic, prob,
                           topic_doc, word_topic, topic_count, alpha/n_topics, beta, step)
            likelihoods.append(-np.mean(sc.gammaln(word_topic + beta).sum(axis=0)
                                        - sc.gammaln(topic_count + vocab_size * beta)))
        plt.figure()
        plt.xlabel('Iterations')
        plt.ylabel('Negative Log Likelihood')
        plt.plot(iterations, likelihoods)
        plt.savefig(plot_file)
    else:
        gibbs_sampling(N, vocab_size, n_topics, doc, word, topic, prob,
                       topic_doc, word_topic, topic_count, alpha/n_topics, beta, n_iter)

    likelihood = np.mean(sc.gammaln(word_topic + beta).sum(axis=0) - sc.gammaln(topic_count + vocab_size * beta))
    word_given_topic = (word_topic + beta) / (topic_count + vocab_size * beta)
    topic_given_doc = (topic_doc + alpha) / (topic_doc.sum(axis=0) + n_topics * alpha)

    return likelihood, word_given_topic, topic_given_doc

@njit
def initialize(N, topic_doc, word_topic, topic_count, doc, word, topic):
    for i in range(N):
        topic_doc[topic[i], doc[i]] += 1
        word_topic[word[i], topic[i]] += 1
        topic_count[topic[i]] += 1


@njit
def gibbs_sampling(N, vocab_size, n_topics, doc, word, topic, prob,
                   topic_doc, word_topic, topic_count, alpha, beta, n_iter):
    # Perform n_iter Gibbs sampling iterations
    for _ in range(n_iter):
        for i in range(N):
            # Compute counts with word i removed from dataset
            topic_doc[topic[i], doc[i]] -= 1
            word_topic[word[i], topic[i]] -= 1
            topic_count[topic[i]] -= 1

            # Compute probability of each topic for word i
            for k in range(n_topics):
                prob[k] = (topic_doc[k, doc[i]] + alpha) * (word_topic[word[i], k] + beta)
                prob[k] /= topic_count[k] + vocab_size * beta
            for k in range(1, n_topics):
                prob[k] += prob[k-1]

            # Sampling
            u = np.random.rand()

            k_low, k_up = -1, n_topics
            while k_up - k_low > 1:
                k_mid = (k_up + k_low) // 2
                if u < prob[k_mid] / prob[n_topics - 1]:
                    k_up = k_mid
                else:
                    k_low = k_mid
            topic[i] = k_up

            # Compute counts with word i included in dataset
            topic_doc[topic[i], doc[i]] += 1
            word_topic[word[i], topic[i]] += 1
            topic_count[topic[i]] += 1


def main():
    mat = sparse.coo_matrix(np.array([[12, 8, 4, 0, 1, 1],
                                      [14, 7, 9, 1, 0, 0],
                                      [18, 9, 6, 1, 1, 1],
                                      [1, 0, 1, 19, 21, 16],
                                      [0, 1, 0, 32, 26, 10]]))

    likelihood, word_given_topic, _ = train(doc_word_mat=mat, n_topics=2, alpha=2, beta=0.01, n_iter=100,
                                            plot_likelihood=False, plot_file='plot.png')

    print(word_given_topic)
    print(likelihood)


if __name__ == "__main__":
    main()