from model import Model
import numpy as np
from numba import njit
import os
import matplotlib.pyplot as plt
import time


def l2_normalize_rows(mat):
    return mat / (1e-30 + np.linalg.norm(mat, axis=1)[:, np.newaxis])


def l1_normalize_rows(mat):
    return mat / (1e-30 + np.sum(mat, axis=1)[:, np.newaxis])


def cosine_similarity(mat1, mat2):
    """
    Computes the cosine similarity between each row of mat1 and each row of mat2.

    Args:
        mat1, mat2: numpy arrays of shape (n,d)

    Returns:
        A numpy array of shape (n,n) whose (i,j)-th entry contains the cosine similarity of mat1[i] and mat2[j].
    """
    return np.matmul(l2_normalize_rows(mat1), l2_normalize_rows(mat2).T)


def sym_kl_divergence(mat1, mat2):
    """
    Computes the negative symmetrized KL divergence between each row of mat1 and each row of mat2.

    Args:
        mat1, mat2: numpy arrays of shape (n,d)

    Returns:
        A numpy array of shape (n,n) whose (i,j)-th entry contains the negative symmetrized KL divergence of mat1[i] and mat2[j].
    """
    n = mat1.shape[0]
    sym_kl = np.zeros((n, n))

    p = l1_normalize_rows(mat1) + 1e-30
    q = l1_normalize_rows(mat2) + 1e-30

    numba_sym_kl_divergence(sym_kl, p, q)

    return -sym_kl


@njit
def numba_sym_kl_divergence(output, p, q):
    n, d = p.shape
    for i in range(n):
        for j in range(n):
            for k in range(d):
                output[i, j] += (p[i, k] - q[j, k]) * np.log2(p[i, k] / q[j, k]) / 2


def plot_topic_self_consistency(model, n_topics, consistency_fn, output_dir):
    fig = plt.figure(figsize=[20, 10])
    n_cols = 5
    n_rows = len(n_topics) // 3 + 1

    axes=[]
    for i, t in enumerate(n_topics):
        model.set_n_topics(t)

        ax = fig.add_subplot(n_rows, n_cols, i + 1, title="{} topics".format(t))
        axes.append(ax)
        cax = ax.matshow(consistency_fn(model.word_topic.T, model.word_topic.T), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(cax, ax=axes)

    plt.savefig(os.path.join(output_dir, f"{model.corpus_name}_{model.name}_{consistency_fn.__name__}_self_consistency.png"))
    plt.close(fig)


def plot_model_self_consistency(models, n_topic, consistency_fn, output_dir):
    fig = plt.figure(figsize=[15, 5])
    n_cols = 3
    n_rows = 1

    axes = []
    for i, model in enumerate(models):
        model.set_n_topics(n_topic)

        ax = fig.add_subplot(n_rows, n_cols, i + 1, title=f"{model.name}")
        axes.append(ax)
        cax = ax.matshow(consistency_fn(model.word_topic.T, model.word_topic.T), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(cax, ax=axes)

    plt.savefig(
        os.path.join(output_dir, f"{models[0].corpus_name}_{n_topic}_topics_{consistency_fn.__name__}_self_consistency.png"))
    plt.close(fig)


def consistency(model1, model2, n_topics, consistency_fn, output_dir, compare_docs=False):
    fig = plt.figure(figsize=[20, 10])
    n_cols = 5
    n_rows = len(n_topics) // 3 + 1

    axes = []
    for i, t in enumerate(n_topics):
        model1.set_n_topics(t)
        model2.set_n_topics(t)

        ax = fig.add_subplot(n_rows, n_cols, i + 1, title="{} topics".format(t))
        axes.append(ax)
        if compare_docs:
            cax = ax.matshow(consistency_fn(model1.topic_doc, model2.topic_doc), interpolation='nearest')
        else:
            cax = ax.matshow(consistency_fn(model1.word_topic.T, model2.word_topic.T), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(cax, ax=axes)
    fig.text(0.5, 0.4, model2.name, ha='center', va='center')
    fig.text(0.02, 0.75, model1.name, ha='center', va='center', rotation='vertical')

    if compare_docs:
        output_file = f"{model1.corpus_name}_{model1.name}_vs_{model2.name}_{consistency_fn.__name__}_docs.png"
    else:
        output_file = f"{model1.corpus_name}_{model1.name}_vs_{model2.name}_{consistency_fn.__name__}_words.png"

    plt.savefig(os.path.join(output_dir, output_file))
    plt.close(fig)


def main():
    base_dir = os.getcwd()
    n_topics = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Plot self-consistency for various number of topics
    output_dir = os.path.join(base_dir, "self_consistency")

    for corpus_name in ["National_Gazette", "Gazette_of_US"]:
        lda = Model(model_name="lda", corpus_name=corpus_name, dir=base_dir)
        kmeans = Model(model_name="k_means", corpus_name=corpus_name, dir=base_dir)
        plsa = Model(model_name="pLSA", corpus_name=corpus_name, dir=base_dir)

        print(f"{corpus_name}: ")
        for consistency_fn in [cosine_similarity, sym_kl_divergence]:
            for model in [lda, kmeans, plsa]:
                print(f"Computing the self-consistency of {model.name} with {consistency_fn.__name__}...", end=' ')
                start = time.perf_counter()
                plot_topic_self_consistency(model, n_topics, consistency_fn, output_dir)
                print("{0:.2f}".format(time.perf_counter() - start))

    # Plot self-consistency for all models with 60 topics

    for corpus_name in ["National_Gazette", "Gazette_of_US"]:
        lda = Model(model_name="lda", corpus_name=corpus_name, dir=base_dir)
        kmeans = Model(model_name="k_means", corpus_name=corpus_name, dir=base_dir)
        plsa = Model(model_name="pLSA", corpus_name=corpus_name, dir=base_dir)

        print(f"{corpus_name}: ")
        for consistency_fn in [cosine_similarity, sym_kl_divergence]:
            print(f"Computing self-consistency with {consistency_fn.__name__}...", end=' ')
            start = time.perf_counter()
            plot_model_self_consistency([kmeans, plsa, lda], 60, consistency_fn, output_dir)
            print("{0:.2f}".format(time.perf_counter() - start))

    # Plot all consistency matrices
    output_dir = os.path.join(base_dir, "consistency")

    for corpus_name in ["National_Gazette", "Gazette_of_US"]:
        lda = Model(model_name="lda", corpus_name=corpus_name, dir=base_dir)
        kmeans = Model(model_name="k_means", corpus_name=corpus_name, dir=base_dir)
        plsa = Model(model_name="pLSA", corpus_name=corpus_name, dir=base_dir)

        print(f"{corpus_name}: ")
        for consistency_fn in [cosine_similarity, sym_kl_divergence]:
            for model1, model2 in [(kmeans, lda), (kmeans, plsa), (plsa, lda)]:
                print(f"Computing the {consistency_fn.__name__} of {model1.name} vs {model2.name}...", end=' ')
                start = time.perf_counter()
                consistency(model1, model2, n_topics, consistency_fn,
                            os.path.join(output_dir, f"{model1.name}_vs_{model2.name}"), compare_docs=False)
                consistency(model1, model2, n_topics, consistency_fn,
                            os.path.join(output_dir, f"{model1.name}_vs_{model2.name}"), compare_docs=True)
                print("{0:.2f}".format(time.perf_counter() - start))


if __name__ == "__main__":
    main()