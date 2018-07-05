from linkers.sparse import SparseLinker
from traceback import format_exc
# import gensim


class DenseLinker(SparseLinker):
    def __init__(self, model_dir, embeddings_fpath, tfidf=True, use_overlap=True, description="", stop_words=True):
        SparseLinker.__init__(self, model_dir, tfidf, use_overlap, description, stop_words)
        self._params["word_embeddings"] = embeddings_fpath
        # load embeddings_fpath

    def link(self, context, phrases):

        # use the loaded embbeddings to compute dense vectors on the fly

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

                    candidate_vectors = self._vectors[indices]
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
                    linked_phrases.append((phrase, best_candidate))
                else:
                    print("Warning: phrase '{}' is not found in the vocabulary of the model".format(phrase))

                    linked_phrases.append((phrase, Candidate()))
            except:
                print("Error while processing phrase '{}':")
                print(format_exc())
                linked_phrases.append((phrase, Candidate()))
        return linked_phrases


 
