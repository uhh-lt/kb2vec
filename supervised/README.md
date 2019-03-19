# kb2vec/supervised 

This project provides an alternative approach to Entity Disambiguation. The input of the feedforward neural network is the concatenation of context vector, span vector, entity graph embeddings, and long abstract (of corresponding entity) vector. 

Installation
-----------

```
cd supervised
pip install -r requirements.txt
```

Set up
-----------

1 -  Creating entity graph embeddings:

From DBpedia datasets (https://wiki.dbpedia.org/develop/datasets/downloads-2016-10), long 
abstracts, labels, and page links files are downloaded. Using `../construct_graph.py`, the graph is contructed.
Page links are the inputs of DeepWalk  algorithm (https://github.com/phanein/deepwalk) to create entity graph embeddings 
Entity graph embedding size is 64.

2 - Candidate entities:
 
We have used the candidate dictionary provided by end2end (https://github.com/dalab/end2end_neural_el), `prob_yago_crosswikis_wikipedia_p_e_m.txt`. 

3 - Preprocess idmaps:

`idmaps` folder contains the files of id2id maps. `chunkid2contextid.txt` consists of context name - index pairs where
context refers to the context in the datasets and index is the index of the context vectors in `vectors/context_vecs.npy` 
for corresponding context. The context vectors are created from pretrained `doc2vec` model using `infer_vector` method.
Similarly, `longabsid2graphid.txt` file contains long abstact vector index and corresponding graph id pairs.
 
Because we use the candidate entity dictionary by end2end, we need to convert their `wiki_id` or `entity id` to our `graph id`. `graphid2wikiid.txt` file contains this map information.
Subset of entity vectors which are used in datasets are saved in `vectors/ent_vecs_graph.npy`. To reach the corresponding vector `wikiid2nnid.txt` map is used.
  
 4 - Training
 
 To train the neural network, create an instance of `InputVecGenerator`. Then, create
 inputs and outputs for train and test by calling its `process` method. With these, 
 have an instance of `Trainer`, and call `train` and `test`, respectively.
