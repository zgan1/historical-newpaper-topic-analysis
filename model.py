import numpy as np
import os
import scipy.sparse as sparse
import pickle


class Model:
    def __init__(self, model_name="", corpus_name="", n_topics=0, dir=""):
        self.name = model_name
        self.corpus_name = corpus_name
        self.n_topics = n_topics
        self.dir = dir

        self.doc_word_matrix = None
        self.word_to_index = None
        self.index_to_word = None
        self.n_docs = 0
        self.vocab_size = 0

        self.load_corpus()

        self.word_topic = None
        self.topic_doc = None
        self.doc_to_topic = None

        if n_topics > 0:
            self.load_model()

    def load_corpus(self):
        path = os.path.join(self.dir, "vocabulary_files", self.corpus_name)

        with open(os.path.join(path, f"{self.corpus_name}_matrix.npz"), "rb") as f:
            self.doc_word_matrix = sparse.load_npz(f)
            self.n_docs, self.vocab_size = self.doc_word_matrix.shape

        with open(os.path.join(path, f"{self.corpus_name}_wi.txt"), "rb") as f:
            self.word_to_index = pickle.loads(f.read())

        with open(os.path.join(path, f"{self.corpus_name}_iw.txt"), "rb") as f:
            self.index_to_word = pickle.loads(f.read())

    def load_model(self):
        with open(os.path.join(self.dir, "models",
                               f"{self.corpus_name}_{self.name}_{self.n_topics}_topics.txt"), "rb") as f:
            self.word_topic = pickle.loads(f.read())
            if self.name == "k_means":
                self.word_topic = self.word_topic.T

        with open(os.path.join(self.dir, "models",
                               f"{self.corpus_name}_{self.name}_{self.n_topics}_topics_doc.txt"), "rb") as f:
            self.topic_doc = pickle.loads(f.read())
            if self.name == "k_means":
                self.doc_to_topic = self.topic_doc
                self.topic_doc = np.zeros((self.n_topics, len(self.doc_to_topic)))
                for d, t in enumerate(self.doc_to_topic):
                    self.topic_doc[t, d] = 1
            else:
                self.doc_to_topic = np.argmax(self.topic_doc, axis=0)

    def change_to(self, new_model_name, new_n_topics=0):
        self.name = new_model_name
        self.set_n_topics(new_n_topics)

    def set_n_topics(self, new_n_topics):
        self.n_topics = new_n_topics

        if new_n_topics > 0:
            self.load_model()
        else:
            self.word_topic = None
            self.topic_doc = None

    def print_info(self):
        print(f"Corpus: {self.corpus_name}")
        print(f"Number of documents: {self.n_docs}")
        print(f"Size of vocabulary: {self.vocab_size}")
        print("")
        if self.n_topics > 0:
            print(f"Model: {self.name}")
            print(f"Number of topics: {self.n_topics}")
            print(f"Word topic matrix: {self.word_topic.shape}")
            print(f"Topic doc matrix: {self.topic_doc.shape}")
            print(f"Doc to topic array: {self.doc_to_topic.shape}")
            print("")

    def docs_containing(self, list_words):
        """
        Returns a numpy array of shape (n_topics,) giving the number of documents in each topic containing all the words
        in list_words.
        """
        freqs = np.zeros(self.n_topics, dtype=np.int)

        indices = [self.word_to_index[w] for w in list_words if w in self.word_to_index]
        n = len(indices)
        if n > 0:
            word_count = np.count_nonzero(self.doc_word_matrix.toarray()[:, indices], axis=1)
            for t in self.doc_to_topic[word_count == n]:
                freqs[t] += 1

        return freqs

