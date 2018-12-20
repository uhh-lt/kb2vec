# kb2vec

Vectorizing knowledge bases for entity linking

Installation
-----------

```
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
```


Start the web service:
---------------------

Entity linking NIF server:

```
python nif_ws.py
```

which will run at ``http://localhost:5000``

GERBIL NIF-based evaluation server (from the ``gerbil`` directory):

```
bash start.sh
```

which will run at ``http://localhost:1234/gerbil``


DBpedia entity linking NIF wrapper (from the ``gerbil-dbpedia-ws`` directory):

```
docker-compose up -d
```

which will run at ``http://localhost:8181/spotlight``


http://localhost:8181/spotlight
http://localhost:5000/random
http://localhost:5000/sparse_overlap
http://localhost:5000/dense_overlap
http://localhost:5000/supertagger
