#from model import Model
from supervised.model import Model
import tensorflow as tf
from supervised.preprocess.prepro_train import InputVecGenerator
import argparse


class Trainer(object):
    def __init__(self, learning_rate=None, h1_size=None, h2_size=None, h3_size=None, h4_size=None,
                 h5_size=None, h6_size=None, l2_beta=None, num_filters=None, cnn=None,
                 device=None, training_epochs=None, dropout=None,
                 train_inputs=None, train_outputs=None, test_inputs=None, test_outputs=None):
        self.model = Model(learning_rate, h1_size, h2_size, h3_size, h4_size,
                           h5_size, h6_size, l2_beta, num_filters, cnn)
        self.device = device
        self.training_epochs = training_epochs
        self.keep_probability = dropout
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs
        self.sess = None

    def train(self):
        self.model.build_model(x_size=self.train_inputs.shape[1], y_size=self.train_outputs.shape[1])

        with tf.device(self.device):
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            self.sess.run(tf.global_variables_initializer())

            for epoch in range(self.training_epochs):
                _, current_loss = self.sess.run([self.model.optimizer, self.model.loss],
                                                feed_dict={self.model.X: self.train_inputs,
                                                           self.model.y: self.train_outputs,
                                                           self.model.keep_prob: self.keep_probability})

    def save_sess(self, path):
        saver = tf.train.Saver()
        if self.sess:
            saver.save(self.sess, path)
        else:
            print('no open session!')

    def close_sess(self):
        self.sess.close()

    def restore_sess(self, path=None):
        if path is None:
            path='/Users/sevgili/Desktop/trained_models/model_aida_train.meta'

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        trained_model = tf.train.import_meta_graph(path, clear_devices=True)
        trained_model.restore(sess, tf.train.latest_checkpoint('/Users/sevgili/Desktop/trained_models/'))

        self.sess = sess
        graph = tf.get_default_graph()

        self.model.pred = graph.get_tensor_by_name('prediction:0')
        self.model.X = graph.get_tensor_by_name('X:0')
        self.model.y = graph.get_tensor_by_name('y:0')
        self.model.keep_prob = graph.get_tensor_by_name('keep_prob:0')

    def test(self):
        correct_prediction = tf.equal(tf.round(self.model.pred), self.model.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Precision, Recall, F1
        TP = tf.count_nonzero(tf.round(self.model.pred) * self.model.y)
        FP = tf.count_nonzero(tf.round(self.model.pred) * (self.model.y - 1))
        FN = tf.count_nonzero((tf.round(self.model.pred) - 1) * self.model.y)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        if train_inputs is not None:
            training_acc = accuracy.eval(session=self.sess, feed_dict={self.model.X: self.train_inputs,
                                                                       self.model.y: self.train_outputs,
                                                                       self.model.keep_prob: self.keep_probability})
            print("Training Accuracy:", training_acc)
            training_precision = precision.eval(session=self.sess, feed_dict={self.model.X: self.train_inputs,
                                                self.model.y: self.train_outputs,
                                                self.model.keep_prob: self.keep_probability})
            training_recall = recall.eval(session=self.sess, feed_dict={self.model.X: self.train_inputs,
                                           self.model.y: self.train_outputs,
                                           self.model.keep_prob: self.keep_probability})
            training_f1 = f1.eval(session=self.sess, feed_dict={self.model.X: self.train_inputs,
                                    self.model.y: self.train_outputs,
                                    self.model.keep_prob: self.keep_probability})
            print("Training Precision:", training_precision, "Training Recall:", training_recall, "Training F1:",
                  training_f1)

        test_acc = accuracy.eval(session=self.sess, feed_dict={self.model.X: self.test_inputs,
                                  self.model.y: self.test_outputs, self.model.keep_prob: 1})
        print("Test Accuracy:", test_acc)
        test_precision = precision.eval(session=self.sess, feed_dict={self.model.X: self.test_inputs,
                                         self.model.y: self.test_outputs, self.model.keep_prob: 1})
        test_recall = recall.eval(session=self.sess, feed_dict={self.model.X: self.test_inputs,
                                   self.model.y: self.test_outputs, self.model.keep_prob: 1})
        test_f1 = f1.eval(session=self.sess, feed_dict={self.model.X: self.test_inputs,
                           self.model.y: self.test_outputs, self.model.keep_prob: 1})
        print("Test Precision:", test_precision, "Test Recall:", test_recall, "Test F1:", test_f1)


def get_parameters():
    parser = argparse.ArgumentParser(
        description='Performs training to discriminate the correct disambiguated entity '
                    'for an ambigous mention of an entity using '
                    'graph embeddings and doc2vec.')

    parser.add_argument('-training_corpus',
                        help="Path for a training corpus.",
                        default='/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/ace2004.txt')

    parser.add_argument('-test_corpus',
                        help="Path for a test corpus.",
                        default='/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/msnbc.txt')

    parser.add_argument('-graph_embedding', help="Path for the pretrained embeddings of a graph "
                                                 "there is an embedding for each node in the entity graph.",
                        default='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/nodes.embeddings')

    parser.add_argument('-graph_entity_vec', help="Path for the subset embeddings of a graph, containing each "
                                                  "candidate graph embeddings.",
                        default='/Users/sevgili/PycharmProjects/end2end_neural_el/data/entities/ent_vecs/ent_vecs_graph.npy')

    parser.add_argument('-doc2vec', help="Path for pretrained doc2vec.",
                        default='/Users/sevgili/Ozge-PhD/wikipedia-doc2vec/all-dim100/wikipedia_document_dim100_with_wikicorpus.doc2vec')

    parser.add_argument('-url2graphid_db',
                        help="Path for the nodes lookup database which is SqliteDict and whose keys denote the "
                             "entities in the graph and values refer to the id of each entity.",
                        default='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db')

    parser.add_argument('-graphid2url_db',
                        help="Path for the nodes lookup database which is SqliteDict and whose keys denote the "
                             "entities in the graph and values refer to the id of each entity.",
                        default='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup_inv.db')

    parser.add_argument('-url2longabs_db', help="Path for the nodes long abstracts database which is SqliteDict "
                                                   "and whose keys denote the node id "
                                                   "and values contain the long abstracts of each entity.",
                        default='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/long_abstracts.db')

    parser.add_argument('-gpu', help="The decision of cpu and gpu, if you would like to use gpu please "
                                     "give only number, e.g. -gpu 0 (default cpu).",
                        default='/cpu')

    parser.add_argument('-learning_rate', help="Learning rate for the optimizer in neural network (default 0.005).",
                        default=0.005, type=float)

    parser.add_argument('-training_epochs',
                        help="The number of epochs to train the neural network (default 15000).",
                        default=15000, type=int)

    parser.add_argument('-h1_size', help="The size of the first hidden layer the neural network (default 100).",
                        default=100, type=int)

    parser.add_argument('-h2_size', help="The size of the second hidden layer the neural network (default 100).",
                        default=100, type=int)

    parser.add_argument('-h3_size', help="The size of the third hidden layer the neural network "
                                         "(default 0, means no third hidden layer!).",
                        default=100, type=int)

    parser.add_argument('-h4_size', help="The size of the fourth hidden layer the neural network "
                                         "(default 0, means no fourth hidden layer!). (Note: if you specify this "
                                         "layer but not previous ones, then all previous layers size become 100)",
                        default=100, type=int)

    parser.add_argument('-h5_size', help="The size of the fifth hidden layer the neural network "
                                         "(default 0, means no fifth hidden layer!). (Note: if you specify this "
                                         "layer but not previous ones, then all previous layers size become 100)",
                        default=0, type=int)

    parser.add_argument('-h6_size', help="The size of the sixth hidden layer the neural network "
                                         "(default 0, means no sixth hidden layer!). (Note: if you specify this "
                                         "layer but not previous ones, then all previous layers size become 100)",
                        default=0, type=int)

    parser.add_argument('-l2_beta', help="The beta parameter of L2 loss (default 0, means no L2 loss!)",
                        default=0.0001, type=float)

    parser.add_argument('-dropout', help="The keep probability of dropout (default 0.5, means dropout!)",
                        default=0.5, type=float)

    parser.add_argument('-num_filters', help="The number of filters for the first CNN layer (default 75)",
                        default=75, type=int)

    parser.add_argument('-cnn', help="The decision whether CNN is included to the architecture or not"
                                     " (default False)",
                        default=False, type=bool)

    parser.add_argument('-file2write', help="Path to write the results (default output.txt)",
                        default='output.txt')

    args = parser.parse_args()

    device = args.gpu
    if device != '/cpu':
        device = '/device:GPU:' + device

    h3_size, h4_size, h5_size = args.h3_size, args.h4_size, args.h5_size
    if args.h6_size != 0 and h5_size == 0:
        h5_size = 100

    if h5_size != 0 and h4_size == 0:
        h4_size = 100

    if h4_size != 0 and h3_size == 0:
        h3_size = 100

    return args.training_corpus, args.test_corpus, args.graph_embedding, args.doc2vec, args.graphid2url_db, \
           args.url2graphid_db, args.url2longabs_db, args.learning_rate, args.training_epochs, args.h1_size, \
           args.h2_size, h3_size, h4_size, h5_size, args.h6_size, device, args.l2_beta, args.dropout, \
           args.num_filters, args.cnn, args.file2write, args.graph_entity_vec


if __name__ == "__main__":
    training_corpus, test_corpus, graph_embedding, doc2vec, graphid2url_db, url2graphid_db, url2longabs_db, \
    learning_rate, training_epochs, h1_size, h2_size, h3_size, h4_size, h5_size, \
    h6_size, device, l2_beta, dropout, num_filters, cnn, file2write, graph_entity_vec = get_parameters()

    input_generator = InputVecGenerator(graph_entity_path=graph_entity_vec, doc2vec_path=doc2vec,
                 url2graphid_db=url2graphid_db, graphid2url_db=graphid2url_db, url2longabs_db=url2longabs_db)
    train_inputs, train_outputs = input_generator.process(path=training_corpus)
    #train_inputs, train_outputs = None, None
    test_inputs, test_outputs = input_generator.process(path=test_corpus)

    print('train size', len(train_inputs), 'test size', len(test_inputs))

    #learning_rate=0.005, training_epochs=15000, h1_size=100, h2_size=100, h3_size=100, h4_size=0,
    #                  h5_size=0, h6_size=0, device='/cpu', l2_beta=0.0001, dropout=0.5, num_filters=75, cnn=False,
    #                  train_inputs=train_inputs, train_outputs=train_outputs, test_inputs=test_inputs, test_outputs=test_outputs

    trainer = Trainer(learning_rate, h1_size, h2_size, h3_size, h4_size, h5_size, h6_size,
                      l2_beta, num_filters, cnn, device, training_epochs, dropout,
                      train_inputs=train_inputs, train_outputs=train_outputs,
                      test_inputs=test_inputs, test_outputs=test_outputs)

    trainer.train()
    trainer.save_sess(path='trained_models/model_msnbc')
    #trainer.restore_sess()
    trainer.test()

