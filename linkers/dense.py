from linkers.sparse import SparseLinker
from traceback import format_exc
from gensim.models import KeyedVectors
import numpy as np
from utils import overlap
from candidate import Candidate
from time import time
from numpy import dot, argmax
from traceback import format_exc
from os.path import exists, join
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from tqdm import tqdm
from candidate import make_phrases
from numpy import argsort, argmax



class DenseLinker(SparseLinker):
    def __init__(self, model_dir, embeddings_fpath, tfidf=True, use_overlap=True, description="", stop_words=True):
        SparseLinker.__init__(self, model_dir, tfidf, use_overlap, description, stop_words)
        self._params["word_embeddings"] = embeddings_fpath
        self._wv = self._load_word_embbeddings(embeddings_fpath)
        self._stopwords = set(stopwords.words("english"))

        print("Normalizing dense vectors...")
        tic = time()
        self._dense_vectors = normalize(self._dense_vectors)
        print("Done in {:.2f} sec.".format(time() - tic))

    def _build_index2candidate(self, candidate2index):
        """ Constructs an index in the opposite direction. """

        index2candidate = {}
        for candidate in candidate2index:
            index = candidate2index[candidate]
            index2candidate[index] = candidate

        return index2candidate

    def print_most_similar(self, n=10, max_candidates=10, test_name="Seal"):
        test_phrases = make_phrases([test_name])

        for test_phrase in test_phrases:
            print("=" * 50, "\n", test_phrase)
            test_candidates = self._phrase2candidates[self._default_phrase(test_phrase)]

            for j, tc in enumerate(test_candidates):
                if j > max_candidates: break

                print("=" * 50, "\n", tc)

                tc_index = self._candidate2index[tc]
                tc_dvector = self._dense_vectors[tc_index, :]

                # dot product with all candidates to find the most similar ones
                tc_sims = self._dense_vectors.dot(tc_dvector)
                tc_sorted_indices = argsort(-tc_sims)[:n]

                print("-" * 50)
                for i, nearest_candidate_index in enumerate(tc_sorted_indices):
                    print(i, tc_sims[nearest_candidate_index], self._index2candidate[nearest_candidate_index], "\n")

    def _load(self, model_dir):
        SparseLinker._load(self, model_dir)

        dense_vectors_filename = "dense_vectors.pkl"
        self._dense_vectors_fpath = join(model_dir, dense_vectors_filename)

        if exists(self._dense_vectors_fpath):
            print("Loading:", self._dense_vectors_fpath)
            self._dense_vectors = joblib.load(self._dense_vectors_fpath)

    def train(self, dataset_fpaths):
        phrases = self._dataset2phrases(dataset_fpaths)
        self._dense_vectors = np.zeros((self._vectors.shape[0], self._wv.vector_size))

        for phrase in tqdm(phrases):
            try:
                dphrase = self._default_phrase(phrase)
                if dphrase in self._phrase2candidates:
                    # get the candidates
                    candidates = list(self._phrase2candidates[dphrase])  # to remove
                    indices = []
                    for candidate in candidates:
                        if candidate in self._candidate2index:
                            indices.append(self._candidate2index[candidate])
                        else:
                            print("Warning: candidate '{}' is not indexed".format(candidate))
                            indices.append(0)  # just to make sure lengths are equal

                    #candidate_vectors = self._vectors[indices]
                    print("Retrieved {} candidates for '{}'".format(len(indices), phrase.text))

                    for index in indices:
                        self._dense_vectors[index, :] = self._get_dense_vector(self._vectors[index, :], dphrase.text)
            except:
                print("Warning: error phrase '{}'".format(phrase))
                print(format_exc())

        joblib.dump(self._dense_vectors, self._dense_vectors_fpath)
        print("Dense vectors:", self._dense_vectors_fpath)

    def _load_word_embbeddings(self, word_embeddings_fpath):
        print("Loading word vectors from:", word_embeddings_fpath)
        tic = time()

        self._params["word_embeddings_pickle"] = word_embeddings_fpath + ".pkl"
        if exists(self._params["word_embeddings_pickle"]):
            wv = KeyedVectors.load(self._params["word_embeddings_pickle"])
            wv.init_sims(replace=True)
        else:
            wv = KeyedVectors.load_word2vec_format(word_embeddings_fpath, binary=False, unicode_errors="ignore")
            wv.init_sims(replace=True)

            tac = time()
            wv.save(self._params["word_embeddings_pickle"])
            print("Saved in {} sec.".format(time() - tac))

        print("Loaded in {} sec.".format(time() - tic))

        return wv

    def link(self, context, phrases):
        linked_phrases = []
        context_vector = self._vectorizer.transform([context])

        for phrase in phrases:
            try:
                dphrase = self._default_phrase(phrase)
                if dphrase in self._phrase2candidates:
                    # get the candidates
                    candidates = list(self._phrase2candidates[dphrase])  # to remove
                    indices = []
                    for candidate in candidates:
                        if candidate in self._candidate2index:
                            indices.append(self._candidate2index[candidate])
                        else:
                            print("Warning: candidate '{}' is not indexed".format(candidate))
                            indices.append(0)  # just to make sure lengths are equal

                    dense_candidate_vectors = self._dense_vectors[indices]
                    # check if candidates are correct
                    print("Retrieved {} candidates for '{}'".format(len(indices), phrase.text))

                    dense_context_vector = self._get_dense_vector(context_vector, dphrase.text)

                    # rank the candidates
                    sims = dot(dense_candidate_vectors, dense_context_vector.T)

                    if self._params["use_overlap"]:
                        overlap_scores = np.zeros(sims.shape)
                        for i, candidate in enumerate(candidates):
                            overlap_scores[i] = overlap(candidate.name, phrase.text)
                    else:
                        overlap_scores = np.ones(sims.shape)

                    scores = np.multiply(sims, overlap_scores)
                    best_index = argmax(scores)
                    best_candidate = candidates[best_index]
                    best_candidate.score = scores[best_index]
                    best_candidate.link = self._get_dbpedia_uri(best_candidate.wiki, best_candidate.uris)
                    linked_phrases.append((phrase, best_candidate))
                else:
                    print("Warning: phrase '{}' is not found in the vocabulary of the model".format(phrase))

                    linked_phrases.append((phrase, Candidate()))
            except:
                print("Error while processing phrase '{}':")
                print(format_exc())
                linked_phrases.append((phrase, Candidate()))
        return linked_phrases

    def _get_dense_vectors(self, sparse_vectors, target):
        dense_vectors = np.zeros((sparse_vectors.shape[0], self._wv.vector_size))

        for i in range(sparse_vectors.shape[0]):
            sparse_candidate_vector = sparse_vectors[i, :]
            dense_candidate_vector = self._get_dense_vector(sparse_candidate_vector, target)
            dense_vectors[i, :] = dense_candidate_vector

        return dense_vectors

    def _get_dense_vector(self, sparse_vector, target):
        """ Construct the dense vector """

        dense_vector = np.zeros(self._wv.vector_size)
        weights_sum = 0.
        names = self._vectorizer.get_feature_names()

        for i, word_weight in enumerate(sparse_vector.data):
            feature_index = sparse_vector.indices[i]
            word = names[feature_index]

            if word.lower() in self._stopwords or word.lower() == target.lower(): continue
            lemma, pos = pos_tag([word])[0]
            if pos[:2] not in ["FW", "JJ", "JJ", "NN", "VB", "RB"]: continue
            # print(word, end=", ")

            if word in self._wv.vocab:
                word_vector = self._wv[word]
            elif word.capitalize() in self._wv.vocab:
                word_vector = self._wv[word.capitalize()]
            else:
                continue


            dense_vector += word_weight * word_vector
            weights_sum += word_weight


        dense_vector = dense_vector / (len(sparse_vector.data) + 1.)
        #print("\n>>>>>>>>\n")
        return dense_vector

 
