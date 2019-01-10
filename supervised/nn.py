import tensorflow as tf
from gensim.models import KeyedVectors, Doc2Vec
import codecs
from sqlitedict import SqliteDict
from nltk.tokenize import RegexpTokenizer
import numpy as np
from random import shuffle
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tensorflow.contrib.tensorboard.plugins import projector


def create_input_from_sample_efficient(path, graph_embeds, doc2vec, lookupdb, longabsdb):
    samples = list()
    tokenizer = RegexpTokenizer(r'\w+')

    with codecs.open(path, "r", "utf-8") as sample:
        line = sample.readline()
        while line != "":
            splitted = line.split('\t')
            line = sample.readline()
            try:
                entity, beg, end, true_url, context, negative_samples_urls = splitted[0], int(splitted[1]), int(splitted[2]), \
                                                                    splitted[3], splitted[4], splitted[5:]
            except:
                continue

            try:
                wordvec = doc2vec[entity]
            except KeyError:
                wordvec = doc2vec.infer_vector(entity)

            contextvec = doc2vec.infer_vector(tokenizer.tokenize(context))

            try:
                longab = longabsdb[true_url]
            except KeyError:
                continue

            try:
                graph_embed = graph_embeds[str(lookupdb[true_url])]
            except KeyError:
                continue
            longabvec = doc2vec.infer_vector(tokenizer.tokenize(longab))

            inputvec = np.concatenate((np.array(wordvec), np.array(graph_embed),
                                       np.array(contextvec), np.array(longabvec)), axis=0)
            # 1 means positive
            samples.append((inputvec, np.array([1])))

            for negative_url in negative_samples_urls:
                negative_url = negative_url.strip()
                try:
                    longab = longabsdb[negative_url]
                except KeyError:
                    continue

                try:
                    graph_embed = graph_embeds[str(lookupdb[negative_url])]
                except KeyError:
                    continue

                longabvec = doc2vec.infer_vector(tokenizer.tokenize(longab))

                inputvec = np.concatenate((np.array(wordvec), np.array(graph_embed),
                                           np.array(contextvec), np.array(longabvec)), axis=0)
                # 0 negative
                samples.append((inputvec, np.array([0])))

    return samples


def format(list_sample):
    inputs, outputs = list(), list()
    index = 0

    for sample in list_sample:
        inputs.insert(index, sample[0])
        outputs.insert(index, sample[1])
        index += 1

    return np.array(inputs), np.array(outputs)


def prepare_training_test_set_efficient(path_samples, graph_embeds, doc2vec, lookupdb, longabsdb):
    samples = create_input_from_sample_efficient(path_samples, graph_embeds, doc2vec, lookupdb, longabsdb)

    shuffle(samples)

    training_size = int(len(samples) * 0.8)
    training_set = samples[:training_size]
    test_set = samples[training_size:]
    del samples

    training_inputs, training_outputs = format(training_set)
    del training_set
    test_inputs, test_outputs = format(test_set)
    del test_set
    return training_inputs, training_outputs, test_inputs, test_outputs


def load_embeds(path_graphembed, path_doc2vec):
    graph_embeds = KeyedVectors.load_word2vec_format(path_graphembed, binary=False)
    print('graph_embeds are loaded')

    doc2vec = Doc2Vec.load(path_doc2vec, mmap = 'r')
    print('doc2vec is loaded')

    return graph_embeds, doc2vec


def load_db(path_lookupdb, path_longabsdb):

    lookupdb = SqliteDict(path_lookupdb, autocommit=False)
    print('lookup is loaded')

    longabsdb = SqliteDict(path_longabsdb, autocommit=False)
    print('long abstract is loaded')

    return lookupdb, longabsdb


def get_parameters():
    parser = argparse.ArgumentParser(description='Performs training to discriminate the correct disambiguated entity '
                                                 'for an ambigous mention of an entity using '
                                                 'graph embeddings and doc2vec.')

    parser.add_argument('-training_corpus',
                        help="Path to a training corpus contaning both positive and negative examples"
                             " in the format -- positive sample, starting location of the ambigous word, "
                             "endinglocation of the ambigous word , true_url, context, negative_samples_urls -- "
                             "seperated by the tabs in text form.",
                        default='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_filtered_randomly_3.tsv')

    parser.add_argument('-graph_embedding', help="Path for the pretrained embeddings of a graph "
                                                 "there is an embedding for each node in the entity graph.",
                        default='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/nodes.embeddings')

    parser.add_argument('-doc2vec', help="Path for pretrained doc2vec.",
                        default='/Users/sevgili/Ozge-PhD/wikipedia-doc2vec/all-dim100/wikipedia_document_dim100_with_wikicorpus.doc2vec')

    parser.add_argument('-lookup_db',
                        help="Path for the nodes lookup database which is SqliteDict and whose keys denote the "
                             "entities in the graph and values refer to the id of each entity.",
                        default='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db')

    parser.add_argument('-long_abstracts_db', help="Path for the nodes long abstracts database which is SqliteDict "
                                                   "and whose keys denote the node id "
                                                   "and values contain the long abstracts of each entity.",
                        default='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/long_abstracts.db')

    parser.add_argument('-gpu', help="The decision of cpu and gpu, if you would like to use gpu please give only number, e.g. -gpu 0"
                                     "(default cpu).",
                        default='/cpu')

    parser.add_argument('-learning_rate', help="Learning rate for the optimizer in neural network (default 0.005).",
                        default=0.005, type=float)

    parser.add_argument('-training_epochs', help="The number of epochs to train the neural network (default 15000).",
                        default=15000, type=int)

    parser.add_argument('-h1_size', help="The size of the first hidden layer the neural network (default 100).",
                        default=100, type=int)

    parser.add_argument('-h2_size', help="The size of the second hidden layer the neural network (default 100).",
                        default=100, type=int)

    parser.add_argument('-h3_size', help="The size of the third hidden layer the neural network "
                                         "(default 0, means no third hidden layer!).",
                        default=0, type=int)

    args = parser.parse_args()

    device = args.gpu
    if device != '/cpu':
        device = '/device:GPU:' + device

    return args.training_corpus, args.graph_embedding, args.doc2vec, args.lookup_db, args.long_abstracts_db, \
           args.learning_rate, args.training_epochs, args.h1_size, args.h2_size, args.h3_size, device


def forward_propagation(x_size, y_size, h1_size, h2_size, h3_size):
    # Weight initializations
    weights = {
        'w1': tf.Variable(tf.random_normal([x_size, h1_size])),
        'w2': tf.Variable(tf.random_normal([h1_size, h2_size])),
        'w3': tf.Variable(tf.random_normal([h2_size, h3_size]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([h1_size])),
        'b2': tf.Variable(tf.random_normal([h2_size])),
        'b3': tf.Variable(tf.random_normal([h3_size])),
        'out': tf.Variable(tf.random_normal([y_size]))
    }

    # Forward propagation
    h1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
    h2 = tf.add(tf.matmul(h1, weights['w2']), biases['b2'])
    h3 = tf.nn.tanh(tf.add(tf.matmul(h2, weights['w3']), biases['b3']))

    # if h3 is specified
    if h3_size:
        weights['out'] = tf.Variable(tf.random_normal([h3_size, y_size]))
        yhat = tf.add(tf.matmul(h3, weights['out']), biases['out'])
    else:
        weights['out'] = tf.Variable(tf.random_normal([h2_size, y_size]))
        yhat = tf.add(tf.matmul(h2, weights['out']), biases['out'])

    return yhat


if __name__ == "__main__":
    training_corpus, path_graphembed, path_doc2vec, path_lookupdb, path_longabsdb, \
    learning_rate, training_epochs, h1_size, h2_size, h3_size, device = get_parameters()

    graph_embeds, doc2vec = load_embeds(path_graphembed, path_doc2vec)
    lookupdb, longabsdb = load_db(path_lookupdb, path_longabsdb)

    inputs, outputs, test_inputs, test_outputs = prepare_training_test_set_efficient(training_corpus, graph_embeds, doc2vec, lookupdb, longabsdb)

    print(inputs.shape, outputs.shape)

    # Input-output layers sizes
    x_size = inputs.shape[1]
    y_size = outputs.shape[1]

    # Symbols
    X = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.float32, shape=[None, y_size])

    # Forward propagation
    yhat = forward_propagation(x_size, y_size, h1_size, h2_size, h3_size)

    # Backward propagation
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yhat, targets=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # -- metrics --
    # Accuracy
    pred = tf.nn.sigmoid(yhat)
    correct_prediction = tf.equal(tf.round(pred), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Precision, Recall, F1
    TP = tf.count_nonzero(tf.round(pred) * y)
    TN = tf.count_nonzero((tf.round(pred) - 1) * (y - 1))
    FP = tf.count_nonzero(tf.round(pred) * (y - 1))
    FN = tf.count_nonzero((tf.round(pred) - 1) * y)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    with tf.device(device):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            loss_summary = tf.summary.scalar('loss', loss)
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)
            precision_summary = tf.summary.scalar('precision', precision)
            recall_summary = tf.summary.scalar('recall', recall)
            f1_summary = tf.summary.scalar('f1_summary', f1)

            config = projector.ProjectorConfig()

            train_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('nn.log', sess.graph)
            projector.visualize_embeddings(summary_writer, config)

            sess.run(tf.global_variables_initializer())

            # adding cross-validation
            kf = KFold(n_splits=5)
            k = 0
            for train_id, val_id in kf.split(inputs, outputs):
                train_inputs = inputs[train_id]
                train_outputs = outputs[train_id]

                val_inputs = inputs[val_id]
                val_outputs = outputs[val_id]

                for epoch in range(training_epochs):
                    _, current_loss = sess.run([optimizer, loss], feed_dict={X: train_inputs, y: train_outputs})

                    if epoch % 500==0 and epoch != 0:
                        #print(current_loss)
                        #print("K is", k)
                        #print("Training Accuracy:", accuracy.eval({X: train_inputs, y: train_outputs}))
                        #print("Precision:", precision.eval({X: train_inputs, y: train_outputs}),
                        #      "Recall:", recall.eval({X: train_inputs, y: train_outputs}),
                        #      "F1:", f1.eval({X: train_inputs, y: train_outputs}))
                        summary_str = sess.run(train_summary_op, feed_dict={X: train_inputs, y: train_outputs})
                        summary_writer.add_summary(summary_str, epoch)

                k += 1
                print("Cross-validation results; acc:", accuracy.eval({X: val_inputs, y: val_outputs}),
                      "precision:", precision.eval({X: val_inputs, y: val_outputs}),
                      "recall:", recall.eval({X: val_inputs, y: val_outputs}),
                      "F1:", f1.eval({X: val_inputs, y: val_outputs}))

            # Test model
            pred = tf.nn.sigmoid(yhat)
            correct_prediction = tf.equal(tf.round(pred), y)

            print("Training Report", classification_report(y_true=outputs,
                                                           y_pred=tf.round(pred).eval({X: inputs, y: outputs})))

            print("Test Report", classification_report(y_true=test_outputs,
                                                       y_pred=tf.round(pred).eval({X: test_inputs, y: test_outputs})))

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Training Accuracy:", accuracy.eval({X: inputs, y: outputs}))
            print("Training Precision:", precision.eval({X: inputs, y: outputs}),
                  "Training Recall:", recall.eval({X: inputs, y: outputs}), "Training F1:", f1.eval({X: inputs, y: outputs}))

            print("Test Accuracy:", accuracy.eval({X: test_inputs, y: test_outputs}))
            print("Test Precision:", precision.eval({X: test_inputs, y: test_outputs}),
                  "Test Recall:", recall.eval({X: test_inputs, y: test_outputs}), " Test F1:",
                  f1.eval({X: test_inputs, y: test_outputs}))


