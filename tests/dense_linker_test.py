from linkers.dense import DenseLinker
from candidate import make_phrases

# embeddings_fpath = "../data/wiki-news-300d-1M.vec"
embeddings_fpath = "../data/crawl-300d-2M.vec"

dataset_fpaths = ["../datasets/dbpedia.ttl.phrases.tsv",
                  "../datasets/kore50.ttl.phrases.tsv",
                  "../datasets/n3-reuters-128.ttl.phrases.tsv"]

l = DenseLinker("../data/count-stopwords-10", embeddings_fpath, stop_words=True, tfidf=False)
l.train(dataset_fpaths)

context = "Madonna is a great music  signer and lives near West Holywood in LA. adonna Louise Ciccone (/tʃɪˈkoʊni/; born August 16, 1958) is an American singer, songwriter, actress, and businesswoman. Referred to as the Queen of Pop since the 1980s, Madonna is known for pushing the boundaries of lyrical content in mainstream popular music, as well as visual imagery in music videos and on stage. She has also frequently reinvented both her music and image while maintaining autonomy within the recording industry. Besides sparking controversy, her works have bee "
phrases = ["Madonna"]

linked_phrases = l.link(context, make_phrases(phrases))
print(linked_phrases)
