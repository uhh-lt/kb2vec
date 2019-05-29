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
import codecs


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
                graph_embed = np.zeros(graph_embeds.vector_size)

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
                    graph_embed = np.zeros(graph_embeds.vector_size)

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

    sample_size = len(samples)
    training_size = int(sample_size * 0.8)
    dev_size = int(sample_size * 0.1)
    training_set = samples[:training_size]
    dev_set = samples[training_size:training_size+dev_size]
    test_set = samples[training_size+dev_size:]
    del samples

    print(sample_size, training_size, dev_size)
    #print(len(training_set), len(dev_set), len(test_set))

    training_inputs, training_outputs = format(training_set)
    del training_set
    dev_inputs, dev_outputs = format(dev_set)
    del dev_set
    test_inputs, test_outputs = format(test_set)
    del test_set
    #print(dev_inputs == dev_inputs)
    return training_inputs, training_outputs, dev_inputs, dev_outputs, test_inputs, test_outputs


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
                        help="Path for a training corpus containing both positive and negative examples"
                             " in the format -- positive sample, starting location of the ambiguous word, "
                             "ending location of the ambiguous word , true_url, context, negative_samples_urls -- "
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

    parser.add_argument('-gpu', help="The decision of cpu and gpu, if you would like to use gpu please "
                                     "give only number, e.g. -gpu 0 (default cpu).",
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

    parser.add_argument('-h4_size', help="The size of the fourth hidden layer the neural network "
                                         "(default 0, means no fourth hidden layer!). (Note: if you specify this "
                                         "layer but not previous ones, then all previous layers size become 100)",
                        default=0, type=int)

    parser.add_argument('-h5_size', help="The size of the fifth hidden layer the neural network "
                                         "(default 0, means no fifth hidden layer!). (Note: if you specify this "
                                         "layer but not previous ones, then all previous layers size become 100)",
                        default=0, type=int)

    parser.add_argument('-h6_size', help="The size of the sixth hidden layer the neural network "
                                         "(default 0, means no sixth hidden layer!). (Note: if you specify this "
                                         "layer but not previous ones, then all previous layers size become 100)",
                        default=0, type=int)

    parser.add_argument('-l2_beta', help="The beta parameter of L2 loss (default 0, means no L2 loss!)",
                        default=0.0, type=float)

    parser.add_argument('-dropout', help="The keep probability of dropout (default 1.0, means dropout!)",
                        default=1.0, type=float)

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

    return args.training_corpus, args.graph_embedding, args.doc2vec, args.lookup_db, args.long_abstracts_db, \
           args.learning_rate, args.training_epochs, args.h1_size, args.h2_size, h3_size, h4_size, h5_size,\
           args.h6_size, device, args.l2_beta, args.dropout, args.num_filters, args.cnn, args.file2write


def forward_propagation(x_size, y_size, h1_size, h2_size, h3_size, h4_size, h5_size, h6_size, l2_beta,
                        cnn, num_filters):
    if cnn:
        # Convolution layer
        filter = tf.Variable(tf.random_normal([num_filters, 1,1]))
        input = tf.expand_dims(X, -1)
        conv = tf.nn.conv1d(input, filter, stride=1, padding='SAME')

        feedforward_input = tf.squeeze(conv, -1)
    else:
        feedforward_input = X

    # Weight initializations
    weights = {
        'w1': tf.Variable(tf.random_normal([x_size, h1_size])),
        'w2': tf.Variable(tf.random_normal([h1_size, h2_size])),
        'w3': tf.Variable(tf.random_normal([h2_size, h3_size])),
        'w4': tf.Variable(tf.random_normal([h3_size, h4_size])),
        'w5': tf.Variable(tf.random_normal([h4_size, h5_size])),
        'w6': tf.Variable(tf.random_normal([h5_size, h6_size]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([h1_size])),
        'b2': tf.Variable(tf.random_normal([h2_size])),
        'b3': tf.Variable(tf.random_normal([h3_size])),
        'b4': tf.Variable(tf.random_normal([h4_size])),
        'b5': tf.Variable(tf.random_normal([h5_size])),
        'b6': tf.Variable(tf.random_normal([h6_size])),
        'out': tf.Variable(tf.random_normal([y_size]))
    }

    # Forward propagation
    h1 = tf.nn.dropout(tf.add(tf.matmul(feedforward_input, weights['w1']), biases['b1']), keep_prob)
    h2 = tf.nn.dropout(tf.add(tf.matmul(h1, weights['w2']), biases['b2']), keep_prob)
    h3 = tf.nn.dropout(tf.add(tf.matmul(h2, weights['w3']), biases['b3']), keep_prob)
    h4 = tf.nn.dropout(tf.add(tf.matmul(h3, weights['w4']), biases['b4']), keep_prob)
    h5 = tf.nn.dropout(tf.add(tf.matmul(h4, weights['w5']), biases['b5']), keep_prob)
    h6 = tf.nn.dropout(tf.add(tf.matmul(h5, weights['w6']), biases['b6']), keep_prob)

    # if h6 is specified
    if h6_size:
        weights['out'] = tf.Variable(tf.random_normal([h6_size, y_size]))
        h6 = tf.nn.tanh(h6)

        yhat = tf.add(tf.matmul(h6, weights['out']), biases['out'])

    # if h5 is specified
    elif h5_size:
        weights['out'] = tf.Variable(tf.random_normal([h5_size, y_size]))
        h5 = tf.nn.sigmoid(h5)

        yhat = tf.add(tf.matmul(h5, weights['out']), biases['out'])

    # if h4 is specified
    elif h4_size:
        weights['out'] = tf.Variable(tf.random_normal([h4_size, y_size]))
        h4 = tf.nn.sigmoid(h4)

        yhat = tf.add(tf.matmul(h4, weights['out']), biases['out'])

    # if h3 is specified
    elif h3_size:
        weights['out'] = tf.Variable(tf.random_normal([h3_size, y_size]))
        h3 = tf.nn.tanh(h3)

        yhat = tf.add(tf.matmul(h3, weights['out']), biases['out'])
    else:
        weights['out'] = tf.Variable(tf.random_normal([h2_size, y_size]))
        yhat = tf.add(tf.matmul(h2, weights['out']), biases['out'])

    l2_loss = 0.0
    if l2_beta:
        l2_loss = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2'])
        if h3_size:
            l2_loss += tf.nn.l2_loss(weights['w3'])
        if h4_size:
            l2_loss += tf.nn.l2_loss(weights['w4'])
        if h5_size:
            l2_loss += tf.nn.l2_loss(weights['w5'])
        if h6_size:
            l2_loss += tf.nn.l2_loss(weights['w6'])
        l2_loss = l2_beta * l2_loss

    return yhat, l2_loss


def train(sess, optimizer, loss, train_inputs, train_outputs, training_epochs, keep_probability):
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        _, current_loss = sess.run([optimizer, loss], feed_dict={X: train_inputs,
                                                                 y: train_outputs, keep_prob:keep_probability})


if __name__ == "__main__":
    training_corpus, path_graphembed, path_doc2vec, path_lookupdb, path_longabsdb, \
    learning_rate, training_epochs, h1_size, h2_size, h3_size, h4_size, h5_size, h6_size, \
    device, l2_beta, dropout, num_filters, cnn, file2write = get_parameters()

    f2write = codecs.open(file2write,  "a", "utf-8")
    f2write.write('NEW RUN \n')
    f2write.write('training_corpus ' + training_corpus + ' learning_rate ' + str(learning_rate) + ' training_epochs ' +
                  str(training_epochs) + ' h1 ' + str(h1_size) + ' h2 ' + str(h2_size) + ' h3 ' + str(h3_size) + ' h4 ' +
                  str(h4_size) + ' h5 ' + str(h5_size) + ' h6 ' + str(h6_size) + ' l2 ' + str(l2_beta) + ' dropout ' +
                  str(dropout) + ' num filters ' + str(num_filters) + '\n')

    print('hidden sizes:', h1_size, h2_size, h3_size, h4_size, h5_size, h6_size)

    graph_embeds, doc2vec = load_embeds(path_graphembed, path_doc2vec)
    lookupdb, longabsdb = load_db(path_lookupdb, path_longabsdb)

    inputs, outputs, dev_inputs, dev_outputs, test_inputs, test_outputs = \
        prepare_training_test_set_efficient(training_corpus, graph_embeds, doc2vec, lookupdb, longabsdb)

    print(inputs.shape, outputs.shape)

    # Input-output layers sizes
    x_size = inputs.shape[1]
    y_size = outputs.shape[1]

    # Symbols
    X = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.float32, shape=[None, y_size])
    keep_prob = tf.placeholder(tf.float32)

    # Forward propagation
    yhat, l2_loss = forward_propagation(x_size, y_size, h1_size, h2_size, h3_size, h4_size,
                                        h5_size, h6_size, l2_beta, cnn, num_filters)

    # Backward propagation
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yhat, targets=y)+l2_loss)
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
        #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            loss_summary = tf.summary.scalar('loss', loss)
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)
            precision_summary = tf.summary.scalar('precision', precision)
            recall_summary = tf.summary.scalar('recall', recall)
            f1_summary = tf.summary.scalar('f1_summary', f1)

            config = projector.ProjectorConfig()

            train_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('nn.log', sess.graph)
            projector.visualize_embeddings(summary_writer, config)

            #sess.run(tf.global_variables_initializer())

            # adding cross-validation
            cv_accuracies = list()
            cv_precisions = list()
            cv_recalls = list()
            cv_f1s = list()

            '''
            kf = KFold(n_splits=5)
            for train_id, val_id in kf.split(inputs, outputs):
                train_inputs = inputs[train_id]
                train_outputs = outputs[train_id]

                val_inputs = inputs[val_id]
                val_outputs = outputs[val_id]

                train(sess=sess, optimizer=optimizer, loss=loss, train_inputs=train_inputs,
                      train_outputs=train_outputs, training_epochs=training_epochs, keep_probability=dropout)

                
                #for epoch in range(training_epochs):
                #    _, current_loss = sess.run([optimizer, loss], feed_dict={X: train_inputs, y: train_outputs})

                #    if epoch % 500==0 and epoch != 0:
                        #print(current_loss)
                        #print("Training Accuracy:", accuracy.eval({X: train_inputs, y: train_outputs}))
                        #print("Precision:", precision.eval({X: train_inputs, y: train_outputs}),
                        #      "Recall:", recall.eval({X: train_inputs, y: train_outputs}),
                        #      "F1:", f1.eval({X: train_inputs, y: train_outputs}))
                #       summary_str = sess.run(train_summary_op, feed_dict={X: train_inputs, y: train_outputs})
                #        summary_writer.add_summary(summary_str, epoch)
                
                cv_accuracy, cv_precision, cv_recall, cv_f1 = accuracy.eval({X: val_inputs, y: val_outputs, keep_prob:1}), \
                                                              precision.eval({X: val_inputs, y: val_outputs, keep_prob:1}), \
                                                              recall.eval({X: val_inputs, y: val_outputs, keep_prob:1}), \
                                                              f1.eval({X: val_inputs, y: val_outputs, keep_prob:1})
                cv_accuracies.append(cv_accuracy)
                cv_precisions.append(cv_precision)
                cv_recalls.append(cv_recall)
                cv_f1s.append(cv_f1)

            print("Cross-validation results; acc:", cv_accuracies, "avg", np.average(cv_accuracies), "std", np.std(cv_accuracies),
                  "precision:", cv_precisions, "avg", np.average(cv_precisions), "std", np.std(cv_precisions),
                  "recall:", cv_recalls, "avg", np.average(cv_recalls), "std", np.std(cv_recalls),
                  "F1:", cv_f1s, "avg", np.average(cv_f1s), "std", np.std(cv_f1s))

            f2write.write("Cross-validation results; acc: " + str(cv_accuracies) + " avg " + str(np.average(cv_accuracies)) +
                          " std " + str(np.std(cv_accuracies)) + " precision: " + str(cv_precisions) + " avg " +
                          str(np.average(cv_precisions)) + " std " + str(np.std(cv_precisions)) + " recall: " +
                          str(cv_recalls) + " avg " + str(np.average(cv_recalls)) + " std " + str(np.std(cv_recalls)) +
                          " F1: " + str(cv_f1s) + " avg " + str(np.average(cv_f1s)) + " std " + str(np.std(cv_f1s)) + '\n')
            '''
            # Train model
            train(sess=sess, optimizer=optimizer, loss=loss, train_inputs=inputs,
                  train_outputs=outputs, training_epochs=training_epochs, keep_probability=dropout)

            # Test model
            pred = tf.nn.sigmoid(yhat)
            correct_prediction = tf.equal(tf.round(pred), y)
            training_report = classification_report(y_true=outputs,
                                                           y_pred=tf.round(pred).eval({X: inputs, y: outputs, keep_prob:dropout}))

            print("Training Report", training_report)
            f2write.write("Training Report " + str(training_report))

            dev_report = classification_report(y_true=dev_outputs,
                                                y_pred=tf.round(pred).eval(
                                                    {X: dev_inputs, y: dev_outputs, keep_prob: 1}))
            f2write.write(" Dev Report " + str(dev_report) + '\n')
            print("Dev Report", dev_report)

            test_report = classification_report(y_true=test_outputs,
                                                       y_pred=tf.round(pred).eval({X: test_inputs, y: test_outputs, keep_prob:1}))
            f2write.write(" Test Report " + str(test_report) + '\n')
            print("Test Report", test_report)

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            training_acc = accuracy.eval({X: inputs, y: outputs, keep_prob:dropout})
            print("Training Accuracy:", training_acc)
            training_precision = precision.eval({X: inputs, y: outputs, keep_prob:dropout})
            training_recall = recall.eval({X: inputs, y: outputs, keep_prob: dropout})
            training_f1 = f1.eval({X: inputs, y: outputs, keep_prob:dropout})
            print("Training Precision:", training_precision, "Training Recall:", training_recall, "Training F1:", training_f1)
            f2write.write("Training Accuracy: " + str(training_acc) + " Training Precision:" + str(training_precision) +
                  " Training Recall: " + str(training_recall)+ " Training F1: " + str(training_f1) + '\n')

            dev_acc = accuracy.eval({X: dev_inputs, y: dev_outputs, keep_prob: 1})
            print("Dev Accuracy:", dev_acc)
            dev_precision = precision.eval({X: dev_inputs, y: dev_outputs, keep_prob: 1})
            dev_recall = recall.eval({X: dev_inputs, y: dev_outputs, keep_prob: 1})
            dev_f1 = f1.eval({X: dev_inputs, y: dev_outputs, keep_prob: 1})
            print("Dev Precision:", dev_precision, "Dev Recall:", dev_recall, "Dev F1:", dev_f1)
            f2write.write("Dev Accuracy: " + str(dev_acc) + " Dev Precision:" + str(dev_precision) +
                          " Dev Recall: " + str(dev_recall) + " Dev F1: " + str(dev_f1) + '\n')

            test_acc = accuracy.eval({X: test_inputs, y: test_outputs, keep_prob:1})
            print("Test Accuracy:", test_acc)
            test_precision = precision.eval({X: test_inputs, y: test_outputs, keep_prob:1})
            test_recall = recall.eval({X: test_inputs, y: test_outputs, keep_prob: 1})
            test_f1 = f1.eval({X: test_inputs, y: test_outputs, keep_prob:1})
            print("Test Precision:", test_precision, "Test Recall:", test_recall, "Test F1:", test_f1)
            f2write.write("Test Accuracy: " + str(test_acc) + " Test Precision:" + str(test_precision) +
                          " Test Recall: " + str(test_recall) + " Test F1: " + str(test_f1) + '\n')
            f2write.close()
