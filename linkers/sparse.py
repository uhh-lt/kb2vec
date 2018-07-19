import numpy as np
from utils import overlap
from linkers.context_aware import ContextAwareLinker 
from candidate import Candidate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from candidate import Phrase, make_phrases
from pandas import read_csv
from time import time
from os.path import join
from utils import ensure_dir
from sklearn.externals import joblib
import json
from os.path import exists
import codecs 
from numpy import dot, argmax
from traceback import format_exc


# ToDo: save also directly the phrase2index file for faster classifications

class SparseLinker(ContextAwareLinker):
    def __init__(self, model_dir, tfidf=True, use_overlap=True, description="", stop_words=True):
        ContextAwareLinker.__init__(self)
        print("Model directory:", model_dir)
        self._params = {}
        self._params["tfidf"] = tfidf
        self._params["description"] = description
        self._params["use_overlap"] = use_overlap
        self._params["stop_words"] = stop_words
        self._params["count_vectorizer_is_binary"] = False
        
        vectorizer_filename = "vectorizer.pkl"
        candidate2index_filename = "candidate2index.pkl"
        params_filename = "params.json"
        vectors_filename = "vectors.pkl"
        phrase2candidates_filename = "phrase2candidates.pkl"
        phrases_filename = "phrases.txt"
        candidates_filename = "candidates.txt"
        
        self._vectorizer_fpath = join(model_dir, vectorizer_filename)
        self._candidate2index_fpath = join(model_dir, candidate2index_filename)
        self._params_fpath = join(model_dir, params_filename) 
        self._vectors_fpath = join(model_dir, vectors_filename)
        self._phrase2candidates_fpath = join(model_dir, phrase2candidates_filename)
        self._phrases_fpath = join(model_dir, phrases_filename)
        self._candidates_fpath = join(model_dir, candidates_filename)
        self._load(model_dir) # using the defined paths

    def set_params(self, params):
        for param in params:
            self._params[param] = params[param]

    def _load(self, model_dir):
        tic = time()
        ensure_dir(model_dir) 

        if exists(self._params_fpath):
            with open(self._params_fpath, "r") as fp:
                self._params = json.load(fp)
            print("Parameters:\n- ", "\n- ".join("{}: {}".format(p, self._params[p]) for p in self._params))
         
        if exists(self._phrase2candidates_fpath):
            print("Loading:", self._phrase2candidates_fpath)
            self._phrase2candidates = joblib.load(self._phrase2candidates_fpath)
        
        if exists(self._candidate2index_fpath):
            print("Loading:", self._candidate2index_fpath)
            self._candidate2index = joblib.load(self._candidate2index_fpath)
        
        if exists(self._vectorizer_fpath):
            print("Loading:", self._vectorizer_fpath)
            self._vectorizer = joblib.load(self._vectorizer_fpath) 
        
        if exists(self._vectors_fpath):
            print("Loading:", self._vectors_fpath)
            self._vectors = joblib.load(self._vectors_fpath)
            
        print("Loaded in {:.2f} sec.".format(time() - tic))

        print("Building index2candidate lookup table...")
        tic = time()
        self._index2candidate = self._build_index2candidate(self._candidate2index)
        print("Done in {:.2f} sec.".format(time() - tic))

    def train(self, dataset_fpaths):
        tic = time()
        print("Training...")
        phrases = self._dataset2phrases(dataset_fpaths) 
        self._train(phrases)
        print("Training is done in {:.2f} sec.".format(time()-tic))
        
    def _train(self, phrases):
        # get the phrases
        with codecs.open(self._phrases_fpath, "w", "utf-8") as out:
            for phrase in phrases: out.write("{}\n".format(phrase.text))
        print("Saved phrases:", self._phrases_fpath)                                      
                                      
        self._params["num_phrases"] = len(phrases)
        print("Number of phrases:", len(phrases))
        
        self._phrase2candidates = self.get_candidates(phrases)

        # get candidates for the phrases
        candidates = set()
        for phrase in self._phrase2candidates:
            for candidate in self._phrase2candidates[phrase]:
                candidates.add(candidate)
        print("Number of candidates:", len(candidates))
        print("Saved phrase2candidate:", self._phrase2candidates_fpath)

        # save the vector indices for the candidates
        with codecs.open(self._candidates_fpath, "w", "utf-8") as out:
            self._candidate2index = {}
            corpus = []
            for index, candidate in enumerate(candidates):
                corpus.append(candidate.text)
                self._candidate2index[candidate] = index

                out.write("{}\t{}\t{}\t{}\n".format(
                    index,
                    candidate.name,
                    candidate.text,
                    "; ".join(candidate.uris)
                ))

            joblib.dump(self._candidate2index, self._candidate2index_fpath)
            print("Saved candidate2index:", self._candidate2index_fpath)
            print("Saved candidates:", self._candidates_fpath)

        # vectorize the text representations of the candidates
        stopwords = 'english' if self._params["stop_words"] else None
        if self._params["tfidf"]:
            self._vectorizer = TfidfVectorizer(stop_words=stopwords)
        else:
            self._vectorizer = CountVectorizer(
                binary=self._params["count_vectorizer_is_binary"],
                stop_words=stopwords)

        self._vectors = self._vectorizer.fit_transform(corpus)
        
        joblib.dump(self._vectorizer, self._vectorizer_fpath) 
        print("Saved vectorizer:", self._vectorizer_fpath)

        joblib.dump(self._vectors, self._vectors_fpath)
        self._params["shape"] = self._vectors.shape
        print("Saved {} candidate feature matrix: {}".format(self._vectors.shape, self._vectors_fpath))

        with open(self._params_fpath, "w") as fp:
            json.dump(self._params, fp)
        print("Saved params:", self._params_fpath)

    def _ttl2phrases(self, ttl_fpaths):
        """ Given a list of ttl files, extract phrases from them. """

        voc = set()
        for dataset_fpath in ttl_fpaths:
            df = read_csv(dataset_fpath, sep="\t", encoding="utf-8")
            for i, row in df.iterrows():
                for target in str(row.targets).split(","):
                    voc.add(target.strip())

        return make_phrases(list(voc))

    def _dataset2phrases(self, dataset_fpaths):
        """ Given a list of datasets, extract phrases from them. """

        voc = set()
        for dataset_fpath in dataset_fpaths:
            df = read_csv(dataset_fpath, sep="\t", encoding="utf-8")
            for i, row in df.iterrows():
                for target in str(row.targets).split(","):
                    voc.add(target.strip())
            
        return make_phrases(list(voc))
       
    def _default_phrase(self, phrase):
        text = phrase.text.strip()
        return Phrase(text, 1, len(text), "http://" + text)

    def link(self, context, phrases):       
        linked_phrases = []
        context_vector = self._vectorizer.transform([context])

        for phrase in phrases:
            try:
                dphrase = self._default_phrase(phrase)
                if dphrase in self._phrase2candidates:
                    # get the candidates
                    candidates = list(self._phrase2candidates[dphrase]) # to remove
                    indices = []
                    for candidate in candidates:
                        if candidate in self._candidate2index:
                            indices.append(self._candidate2index[candidate])
                        else:
                            print("Warning: candidate '{}' is not indexed".format(candidate))
                            indices.append(0) # just to make sure lengths are equal

                    candidate_vectors = self._vectors[ indices ]
                    print("Retrieved {} candidates for '{}'".format(len(indices), phrase.text))

                    # rank the candidates
                    sims = dot(candidate_vectors, context_vector.T)
                    
                    if self._params["use_overlap"]:
                        overlap_scores = np.zeros(sims.shape) 
                        for i, candidate in enumerate(candidates):
                            overlap_scores[i] = overlap(candidate.name, phrase.text)
                    else:
                        overlap_scores = np.ones(sims.shape)

                    scores = np.multiply(sims.toarray(), overlap_scores)
                    best_index = argmax(scores)
                    best_candidate = candidates[best_index]
                    best_candidate.score = scores[best_index][0]
                    best_candidate.link = self._get_dbpedia_uri(best_candidate.wiki, best_candidate.uris)
                    linked_phrases.append( (phrase, best_candidate) )
                else:
                    print("Warning: phrase '{}' is not found in the vocabulary of the model".format(phrase))

                    linked_phrases.append( (phrase, Candidate()) )  
            except:
                print("Error while processing phrase '{}':")
                print(format_exc())
                linked_phrases.append( (phrase, Candidate()) )
        return linked_phrases

