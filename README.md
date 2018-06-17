# kb2vec

Vectorizing knowledge bases for entity linking

Installation
-----------

```
pip install -r requirements.txt
```


Start the web service:
---------------------

Entity linking NIF server:

```
python nif_ws.py
```


GERBIL NIF-based evaluation server (from the ``gerbil`` directory):

```
start.sh
```


DBpedia entity linking NIF wrapper (from the ``gerbil-dbpedia-ws`` directory):

```
docker-compose up -d
```
