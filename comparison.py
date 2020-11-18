import k_means
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def read_common_words(file):
    common_words = []
    with open(file, "r") as f:
        for line in f:
            common_words.append(line.split())
    return common_words


def plsa_vs_k_means(corpus_name, plsa_model_file, plsa_top_words_file, k_means_model_file, k_means_top_words_file, output_dir):
    with open(plsa_model_file, 'rb') as f:
        probability_matrix = pickle.loads(f.read())  # shape = (vocab size, n_topics)

    with open(k_means_model_file, 'rb') as f:
        centroids = pickle.loads(f.read())  # shape = (n_topics, vocab size)

    probability_matrix = k_means.normalize_rows(probability_matrix.T)

    # Compute the similarity between each pLSA topic and each k-means topic, and plot the result
    similarity_matrix = np.matmul(probability_matrix, centroids.T)
    n_topics = similarity_matrix.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(title='Similarity between pLSA and k-means topics',
                         xlabel='k-means', ylabel='pLSA')
    cax = ax.matshow(similarity_matrix, interpolation='nearest')
    fig.colorbar(cax)
    plt.savefig(os.path.join(output_dir,
                             corpus_name + '_plsa_vs_k_means_' + str(n_topics) + '_topics_similarity_plot.png'))

    # For each pLSA topic, find the closest k-means topic and compare their most common words
    pLSA_to_k_means = np.argmax(similarity_matrix, axis=1)
    pLSA_words = read_common_words(plsa_top_words_file)
    k_means_words = read_common_words(k_means_top_words_file)

    with open(os.path.join(output_dir, corpus_name + '_plsa_vs_k_means_' + str(n_topics) + '_topics.tex'), "w") as tex:
        for i in range(n_topics):
            j = pLSA_to_k_means[i]

            tex.write('\\begin{center}')
            table_code = '\\begin{tabularx}{\\textwidth} {\n' \
                         + '  | c | >{\\raggedright\\arraybackslash}X | } \\hline \n'
            tex.write(table_code)
            tex.write('pLSA topic %d' % i + ' & ')
            for word in pLSA_words[i]:
                if word in k_means_words[j]:
                    tex.write('\\textbf{' + word + '} ')
                else:
                    tex.write('\\textcolor{red}{' + word + '} ')
            tex.write('\\\\ \\hline \n')
            tex.write('k-means topic %d' % j + ' & ')
            for word in k_means_words[j]:
                if word in pLSA_words[i]:
                    tex.write('\\textbf{' + word + '} ')
                else:
                    tex.write('\\textcolor{red}{' + word + '} ')
            tex.write('\\\\ \\hline \n')
            tex.write('\\end{tabularx}\n\n')
            tex.write('\\end{center}\n\n')


def plsa_vs_lda(corpus_name, plsa_model_file, plsa_top_words_file, lda_model_file, lda_top_words_file, output_dir):
    with open(plsa_model_file, 'rb') as f:
        plsa_word_topic = pickle.loads(f.read())  # shape = (vocab size, n_topics)

    with open(lda_model_file, 'rb') as f:
        lda_word_topic = pickle.loads(f.read())  # shape = (n_topics, vocab size)

    # Compute the symmetric KL divergence between each pair of pLSA and LDA topics
    p = plsa_word_topic[:, :, np.newaxis] + 1e-15
    q = lda_word_topic[:, np.newaxis, :] + 1e-15
    similarity_matrix = np.sum(p * np.log2(p / q) + q * np.log2(q / p), axis=0) / 2
    n_topics = similarity_matrix.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(title='Symmetric KL divergence between pLSA and LDA topics',
                         xlabel='LDA', ylabel='pLSA')
    cax = ax.matshow(-similarity_matrix, interpolation='nearest')
    fig.colorbar(cax)
    plt.savefig(os.path.join(output_dir,
                             corpus_name + '_plsa_vs_lda_' + str(n_topics) + '_topics_similarity_plot.png'))

    # For each pLSA topic, find the closest LDA topic and compare their most common words
    pLSA_to_lda = np.argmin(similarity_matrix, axis=1)
    pLSA_words = read_common_words(plsa_top_words_file)
    lda_words = read_common_words(lda_top_words_file)

    with open(os.path.join(output_dir, corpus_name + '_plsa_vs_lda_' + str(n_topics) + '_topics.tex'), "w") as tex:
        for i in range(n_topics):
            j = pLSA_to_lda[i]

            tex.write('\\begin{center}')
            table_code = '\\begin{tabularx}{\\textwidth} {\n' \
                         + '  | c | >{\\raggedright\\arraybackslash}X | } \\hline \n'
            tex.write(table_code)
            tex.write('pLSA topic %d' % i + ' & ')
            for word in pLSA_words[i]:
                if word in lda_words[j]:
                    tex.write('\\textbf{' + word + '} ')
                else:
                    tex.write('\\textcolor{red}{' + word + '} ')
            tex.write('\\\\ \\hline \n')
            tex.write('LDA topic %d' % j + ' & ')
            for word in lda_words[j]:
                if word in pLSA_words[i]:
                    tex.write('\\textbf{' + word + '} ')
                else:
                    tex.write('\\textcolor{red}{' + word + '} ')
            tex.write('\\\\ \\hline \n')
            tex.write('\\end{tabularx}\n\n')
            tex.write('\\end{center}\n\n')


def main(n_topics=[5], n_common_words=20, corpus_name='',
         want_plsa_vs_k_means=False,
         want_plsa_vs_lda=False):

    base_dir = os.getcwd()
    model_dir = 'models'
    top_words_dir = 'top_words'
    output_dir = os.path.join(base_dir, 'comparisons')

    pLSA_model_file = [os.path.join(base_dir, model_dir,
                                    corpus_name + '_pLSA_' + str(t) + '_topics.txt') for t in n_topics]
    k_means_model_file = [os.path.join(base_dir, model_dir,
                                       corpus_name + '_k_means_' + str(t) + '_topics.txt') for t in n_topics]
    lda_model_file = [os.path.join(base_dir, model_dir,
                                   corpus_name + '_lda_' + str(t) + '_topics.txt') for t in n_topics]
    top_pLSA_words_file = [os.path.join(base_dir, top_words_dir,
                                        corpus_name + '_pLSA_' + str(t)
                                        + '_topics_' + str(n_common_words) + '_words.txt') for t in n_topics]
    top_k_means_words_file = [os.path.join(base_dir, top_words_dir,
                                           corpus_name + '_k_means_' + str(t)
                                           + '_topics_' + str(n_common_words) + '_words.txt') for t in n_topics]
    top_lda_words_file = [os.path.join(base_dir, top_words_dir,
                                       corpus_name + '_lda_' + str(t)
                                       + '_topics_' + str(n_common_words) + '_words.txt') for t in n_topics]

    if want_plsa_vs_k_means:
        for i in range(len(n_topics)):
            plsa_vs_k_means(corpus_name, pLSA_model_file[i], top_pLSA_words_file[i],
                            k_means_model_file[i], top_k_means_words_file[i], output_dir)

    if want_plsa_vs_lda:
        for i in range(len(n_topics)):
            plsa_vs_lda(corpus_name, pLSA_model_file[i], top_pLSA_words_file[i],
                            lda_model_file[i], top_lda_words_file[i], output_dir)


if __name__ == "__main__":
    main(corpus_name='Pennsylvania_Gazette', n_topics=[40], n_common_words=20,
         want_plsa_vs_k_means=False,
         want_plsa_vs_lda=True)
