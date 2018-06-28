
print("-"*50)
c = candidates[17]
print(c.text, "\n")
index = self._candidate2index[c]
candidate_vector = self._vectors[index]
names = self._vectorizer.get_feature_names()
for i, value in enumerate(candidate_vector.data):
    print(i, candidate_vectors.indices[i], names[candidate_vectors.indices[i]], value)
    
    
get the _vectors and see if we can reconstruct it

print("-"*50)
v = self._vectors[59]
names = self._vectorizer.get_feature_names()
for i, value in enumerate(v.data):
    print(i,
          v.indices[i],
          names[v.indices[i]],
          value)


print("-"*50)
names = self._vectorizer.get_feature_names()
for i, value in enumerate(context_vector.data):
    print(i, context_vector.indices[i], names[context_vector.indices[i]], value)

# print all candidates and their scores
scored_candidates = []
for i, c in enumerate(candidates):
    s = scores[i][0]
    overlap_s = overlap_scores[i][0]
    sim_s = sims[i].toarray()[0][0]
    scored_candidates.append( (s, overlap_s, sim_s, c) )
    
i = 0
for s, overlap_s, sim_s, c in sorted(scored_candidates, reverse=True):
    i += 1
    print("\n * {} {:.2f} {:.2f} {:.2f} '{}' >>> {}".format(i, s, overlap_s, sim_s, c.name, c.text)) 
