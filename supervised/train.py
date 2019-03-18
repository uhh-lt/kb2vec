from model import Model
import tensorflow as tf
from preprocess.prepro_train import InputVecGenerator


class Trainer(object):
    def __init__(self, learning_rate, training_epochs, h1_size, h2_size, h3_size, h4_size, h5_size, h6_size, \
                 device, l2_beta, dropout, num_filters, cnn, train_inputs, train_outputs, test_inputs, test_outputs):
        self.model = Model(learning_rate, h1_size, h2_size, h3_size, h4_size, h5_size, h6_size, l2_beta,
                           num_filters, cnn)
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

    def test(self):
        pred = tf.nn.sigmoid(self.model.yhat)
        correct_prediction = tf.equal(tf.round(pred), self.model.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Precision, Recall, F1
        TP = tf.count_nonzero(tf.round(pred) * self.model.y)
        FP = tf.count_nonzero(tf.round(pred) * (self.model.y - 1))
        FN = tf.count_nonzero((tf.round(pred) - 1) * self.model.y)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

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


if __name__ == "__main__":
    input_generator = InputVecGenerator()
    train_inputs, train_outputs = input_generator.process(
                        path='/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/aida_train.txt')
    test_inputs, test_outputs = input_generator.process(
                        path='/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/msnbc.txt')

    print('train size', len(train_inputs), 'test size', len(test_inputs))

    trainer = Trainer(learning_rate=0.005, training_epochs=15000, h1_size=100, h2_size=100, h3_size=100, h4_size=0,
                      h5_size=0, h6_size=0, device='/cpu', l2_beta=0.0001, dropout=0.5, num_filters=75, cnn=False,
                      train_inputs=train_inputs, train_outputs=train_outputs, test_inputs=test_inputs, test_outputs=test_outputs)

    trainer.train()
    trainer.test()

