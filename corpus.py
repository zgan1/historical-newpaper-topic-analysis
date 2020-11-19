import numpy as np
import nltk
import pickle
import scipy.sparse as sparse
import pkg_resources
import time
from symspellpy import SymSpell, Verbosity


class Corpus:
    """
    This class preprocesses a corpus in the form of txt files to generate
    a vocabulary and a document word matrix.
    """
    def __init__(self, raw_corpus, corpus_file, stopwords_file, vocab_file, stems_file, matrix_file):
        self.raw_corpus = raw_corpus
        self.corpus_file = corpus_file
        self.stopwords_file = stopwords_file
        self.vocab_file = vocab_file
        self.stems_file = stems_file
        self.matrix_file = matrix_file
        self.corpus = []
        self.index_to_word = {}
        self.word_to_index = {}
        self.word_count_mat = None
        self.stopwords = []
        self.stems = {}
        self.corpus_size = 0
        self.vocab_size = 0

    def create_corpus(self):
        """
        Creates a corpus from a list of files.
        """
        for file_path in self.raw_corpus:
            with open(file_path, encoding='utf8') as f_input:
                doc = nltk.wordpunct_tokenize(f_input.read())
                doc = [word.lower() for word in doc if word.isalpha()]
                self.corpus.append(doc)
        self.corpus_size = len(self.corpus)
        with open(self.corpus_file,  'w', encoding="utf-8") as f:
            for line in self.corpus:
                f.write(' '.join(line) + '\n')

        print("done.")

    def spellchecking(self, raw_vocab):
        """
        :param raw_vocab: a set raw vocabularies to be preprocessed.
        :return: a dictionary from the raw vocabulary to the corrected vocabulary.
        """
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        raw_to_correct = {}
        print("start spellcheck timing")
        start = time.perf_counter()
        for element in raw_vocab:
            input_word = element
            suggestions = sym_spell.lookup(input_word, Verbosity.TOP, include_unknown=False,
                                           max_edit_distance=2)
            if len(suggestions) > 0:
                raw_to_correct[input_word] = suggestions[0].term
            else:
                raw_to_correct[input_word] = ""
        end = time.perf_counter()
        print("end timing: ", end-start)
        return raw_to_correct

    def get_raw_vocabulary(self):
        """
        Gets the raw vocabulary of a corpus.
        """
        raw_vocab = set()
        for document in self.corpus:
            for word in document:
                raw_vocab.add(word)
        print("The size of the raw vocabulary is ", len(raw_vocab))
        return raw_vocab

    def load_stopwords(self):
        with open(self.stopwords_file) as stopwords:
            self.stopwords = stopwords.read().split()

    def preprocess(self):
        """
        preprocess a corpus to get the vocabulary of a corpus.
        """
        raw_vocab = self.get_raw_vocabulary()
        raw_to_correct = self.spellchecking(raw_vocab)
        self.get_vocabulary(self.stopwords, raw_to_correct)

    def get_vocabulary(self, stopwords, raw_to_correct):
        """
        Performs stemming and frequency capping for a vocabulary. Generates a index to word dictionary
        and a word to index dictionary.
        :param stopwords: a list of stopwords
        :param raw_to_correct: a raw vocabulary to corrected vocabulary dictionary
        """
        frequencies = {}
        porter = nltk.PorterStemmer()
        for document in self.corpus:
            for word in document:
                if word not in self.stems:
                    converted_word = raw_to_correct[word]
                    self.stems[word] = porter.stem(converted_word)

        with open(self.stems_file, 'wb') as f:
            pickle.dump(self.stems_file, f)

        for document in self.corpus:
            for word in document:
                stem_word = self.stems[word]
                if stem_word and (len(stem_word) > 2) and (stem_word not in stopwords):
                    if stem_word in frequencies:
                        frequencies[stem_word] += 1
                    else:
                        frequencies[stem_word] = 1

        index = 0
        for word, freq in frequencies.items():
            if freq >= 6:
                self.word_to_index[word] = index
                index += 1

        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.vocab_size = len(self.word_to_index)

    def text_to_matrix(self):
        """
        Converts a preprocessed vocabulary and a corpus into a document word matrix.
        """
        self.word_count_mat = np.zeros((self.corpus_size, self.vocab_size), np.int)

        for i, doc in enumerate(self.corpus):
            for word in doc:
                stem = self.stems[word]
                if stem in self.word_to_index:
                    self.word_count_mat[i, self.word_to_index[stem]] += 1

    def read_corpus(self):
        corpus = []
        with open(self.corpus_file, encoding="utf-8") as f:
            for line in f:
                corpus.append(line.split())
        self.corpus = corpus
        self.corpus_size = len(self.corpus)

    def read_vocabulary(self):
        with open(self.vocab_file, "rb") as f:
            self.word_to_index = pickle.loads(f.read())
        self.vocab_size = len(self.word_to_index)

    def read_stems(self):
        with open(self.stems_file, "rb") as f:
            self.stems = pickle.loads(f.read())

    def save_vocabulary(self):
        with open(self.vocab_file, 'wb') as f:
            pickle.dump(self.word_to_index, f)

    def save_document_word_matrix(self):
        sparse.save_npz(self.matrix_file, sparse.coo_matrix(self.word_count_mat))
