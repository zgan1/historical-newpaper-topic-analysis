import numpy as np
import os
import time
import pickle
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def top_words(n_words, word_given_topic):
    return np.argsort(-word_given_topic.T, axis=1)[:, 0:n_words]


def load_word_given_topic(model_name, file):
    with open(file, 'rb') as f:
        word_given_topic = pickle.loads(f.read())
        if model_name == "k_means":
            word_given_topic = word_given_topic.T
    return word_given_topic


def topic_coherence(doc_word_matrix, n_words, topic_words):
    n_topics, n_top_words = topic_words.shape

    doc_word = doc_word_matrix.toarray()

    doc_frequency = np.count_nonzero(doc_word, axis=0)[topic_words]

    co_doc_frequency = np.zeros((n_topics, n_top_words, n_top_words))
    for t in range(n_topics):
        for j in range(1, n_top_words):
            for i in range(j):
                co_doc_frequency[t, i, j] = np.count_nonzero(np.logical_and(doc_word[:, topic_words[t, i]] > 0,
                                                                            doc_word[:, topic_words[t, j]] > 0))
    scores = np.log((co_doc_frequency + 1) / doc_frequency[:, :, np.newaxis])

    coherence = np.empty((n_topics, len(n_words)))
    for ind, n in enumerate(n_words):
        for t in range(n_topics):
            total_score = 0
            for j in range(1, n):
                for i in range(j):
                    total_score += scores[t, i, j]
            coherence[t, ind] = total_score

    return coherence


def plot_coherence_scores_distribution(n_common_words, model_names, n_topics, corpus_names):
    base_dir = os.getcwd()
    vocab_dir = 'vocabulary_files'
    model_dir = 'models'
    output_dir = 'coherence'

    for corpus_name in corpus_names:
        doc_word_file = os.path.join(base_dir, vocab_dir, corpus_name, corpus_name + '_matrix.npz')

        with open(doc_word_file, 'rb') as f:
            doc_word_matrix = sparse.load_npz(f)

        for model_name in model_names:
            q1_coherence_scores = np.empty((len(n_topics), len(n_common_words)))
            median_coherence_scores = np.empty((len(n_topics), len(n_common_words)))
            q3_coherence_scores = np.empty((len(n_topics), len(n_common_words)))
            coherence_scores = []
            for i, t in enumerate(n_topics):
                model_file = os.path.join(base_dir, model_dir,
                                          corpus_name + '_' + model_name + '_' + str(t) + '_topics.txt')
                word_given_topic = load_word_given_topic(model_name, model_file)

                topic_top_words = top_words(np.max(n_common_words), word_given_topic)

                print(f"Computing coherence scores for {corpus_name} with {t} {model_name} topics...", end=' ')
                start = time.perf_counter()
                coherence_scores.append(topic_coherence(doc_word_matrix, n_common_words, topic_top_words))
                print("done. {0:.2f} s".format(time.perf_counter() - start))

                q1_coherence_scores[i, :] = np.quantile(coherence_scores[-1], 0.25, axis=0)
                median_coherence_scores[i, :] = np.median(coherence_scores[-1], axis=0)
                q3_coherence_scores[i, :] = np.quantile(coherence_scores[-1], 0.75, axis=0)

            for i, n in enumerate(n_common_words):
                plt.figure()
                plt.xlabel("Number of topics")
                plt.xticks(n_topics)
                plt.ylabel("Coherence score")
                for j, t in enumerate(n_topics):
                    plt.plot(np.full(t, t), coherence_scores[j][:, i], 'bo')
                plt.plot(n_topics, q3_coherence_scores[:, i], label="Third quartile")
                plt.plot(n_topics, median_coherence_scores[:, i], label="Median")
                plt.plot(n_topics, q1_coherence_scores[:, i], label="First quartile")
                plt.legend()
                plt.savefig(os.path.join(base_dir, output_dir, f"{corpus_name}_{model_name}_top_{n}_words_coherence.png"))
                plt.close()

                with open(os.path.join(base_dir, output_dir,
                                       f"{corpus_name}_{model_name}_top_{n}_words_coherence.txt"), "wb") as f:
                    pickle.dump([coherence_scores[j][:, i] for j, _ in enumerate(n_topics)], f)


def plot_number_coherent_topics(corpus_names, model_names, n_common_words):
    base_dir = os.getcwd()
    output_dir = 'coherence'

    for corpus_name in corpus_names:
        for model_name in model_names:
            for n in n_common_words:
                with open(os.path.join(base_dir, output_dir,
                                       f"{corpus_name}_{model_name}_top_{n}_words_coherence.txt"), "rb") as f:
                    coherence_scores = pickle.loads(f.read())

                n_topics = [s.shape[0] for s in coherence_scores]

                mid = len(coherence_scores) // 2
                thresholds = [np.quantile(coherence_scores[mid], 0.15 * i) for i in range(1, 7)]

                plt.figure()
                plt.xlabel("Total number of topics")
                plt.xticks(n_topics)
                plt.ylabel("Number of topics with coherence above threshold")
                for threshold in thresholds:
                    plt.plot(n_topics, [np.count_nonzero(s > threshold) for s in coherence_scores],
                             label="Threshold = {0:.1f}".format(threshold))
                plt.legend()
                plt.savefig(os.path.join(base_dir, output_dir,
                                         f"{corpus_name}_{model_name}_topics_above_threshold.png"))
                plt.close()


def main():
    plot_coherence_scores_distribution(n_common_words=[30],
                                       model_names=["k_means", "lda", "pLSA"],
                                       n_topics=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
                                       corpus_names=["National_Gazette", "Gazette_of_US"])
    plot_number_coherent_topics(corpus_names=["National_Gazette", "Gazette_of_US"],
                                model_names=["k_means", "lda", "pLSA"],
                                n_common_words=[30])


if __name__ == "__main__":
    main()


