dense_context_vector = np.zeros(self._wv.vector_size)

for i, value in enumerate(context_vector.data):
    feature_index = context_vector.indices[i]
    names = self._vectorizer.get_feature_names()
    
    word = names[feature_index]
    word_weight = value
    
    if word in self._wv.vocab:
        print(word)
        word_vector = self._wv[word]
        dense_context_vector += word_vector
    elif word.capitalize() in self._wv.vocab:
        print(">>>", word.capitalize())
        word_vector = self._wv[word.capitalize()]
        dense_context_vector += word_vector
        
dense_context_vector = dense_context_vector / (i + 1.)




dense_candidate_vectors = np.zeros((candidate_vectors.shape[0],
                                   self._wv.vector_size))

for i in range(candidate_vectors.shape[0]):
    sparse_candidate_vector = candidate_vectors[i, :]
    dense_candidate_vector = self._get_dense_vector(sparse_candidate_vector)
    print(i, dense_candidate_vector)
    
    break
