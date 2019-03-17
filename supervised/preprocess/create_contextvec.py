from prepro_util import Chunker
import util
from gensim.models import Doc2Vec
import numpy as np


def load_doc2vec(path = '/Users/sevgili/Ozge-PhD/wikipedia-doc2vec/all-dim100/wikipedia_document_dim100_with_wikicorpus.doc2vec'):
    return Doc2Vec.load(path, mmap='r')


class ContextVecCreator(object):
    def __init__(self):
        self.doc2vec = load_doc2vec()
        self.url2graphid = util.load_url2graphid()
        self.chunker = Chunker()
        self.contextid2chunkid_file = open('idmaps/contextid2chunkid.txt', 'a')
        self.chunkid2contextid_file = open('idmaps/chunkid2contextid.txt', 'a')

    def chunk2contextvec(self, chunk, context_id):
        chunk_id, chunk_words, begin_gm, end_gm, ground_truth = chunk
        context = ' '.join(chunk_words)
        context_vec = self.doc2vec.infer_vector(context)

        self.contextid2chunkid_file.write(str(context_id) + '\t' + str(chunk_id) + '\n')
        self.chunkid2contextid_file.write(str(chunk_id) + '\t' + str(context_id) + '\n')

        return context_vec

    # create context vec for each context in dataset:
    def create_contextvec(self, dataset_file_paths, dataset_ttl_paths):
        context_embeds = list()
        context_id = 0

        for path in dataset_file_paths:
            for chunk in self.chunker.process(path):
                context_vec = self.chunk2contextvec(chunk, context_id)

                context_embeds.append(context_vec)
                context_id += 1

        for path in dataset_ttl_paths:
            for chunk in self.chunker.process_ttl(path, self.url2graphid):
                context_vec = self.chunk2contextvec(chunk, context_id)

                context_embeds.append(context_vec)
                context_id += 1

        np.save(file='vectors/context_vecs.npy', arr=np.array(context_embeds))


if __name__ == "__main__":
    dataset_file_paths=['/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/ace2004.txt',
                                          '/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/aida_dev.txt',
                                          '/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/aida_test.txt',
                                          '/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/aida_train.txt',
                                          '/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/aquaint.txt',
                                          '/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/clueweb.txt',
                                          '/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/msnbc.txt']
    dataset_ttl_paths=['/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/dbpedia-spotlight-nif.ttl',
                       '/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/kore50-nif.ttl',
                       '/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl',
                       '/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/Reuters-128.ttl',
                       '/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/RSS-500.ttl']

    creator = ContextVecCreator()
    creator.create_contextvec(dataset_file_paths, dataset_ttl_paths)

