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

2 - Preprocess idmaps:

`idmaps` folder contains the files of id2id maps. `chunkid2contextid.txt` consists of context name - index pairs where
 context refers to the context in the datasets and index is the index of the context vectors in `vectors/context_vecs.npy` 
 for corresponding context. The context vectors are created from pretrained `doc2vec` model using `infer_vec` method.
 
 3 - Training
 
 To train the neural network, create an instance of `InputVecGenerator`. Then, create
 inputs and outputs for train and test by calling its `process` method. With these, 
 have an instance of `Trainer`, and call `train` and `test`, respectively.
