import tensorflow as tf


class Model(object):
    def __init__(self, learning_rate, h1_size, h2_size, h3_size, h4_size, h5_size, h6_size, \
                 l2_beta, num_filters, cnn):
        self.lr = learning_rate
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.h3_size = h3_size
        self.h4_size = h4_size
        self.h5_size = h5_size
        self.h6_size = h6_size
        self.l2_beta = l2_beta
        self.num_filters = num_filters
        self.cnn = cnn
        self.X = None
        self.y = None
        self.keep_prob = None
        self.yhat = None
        self.loss = None
        self.optimizer = None
        self.pred = None

    def create_placeholders(self, x_size, y_size):
        self.X = tf.placeholder(tf.float32, shape=[None, x_size], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, y_size], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def forward_propagation(self, x_size, y_size):
        if self.cnn:
            # Convolution layer
            filter = tf.Variable(tf.random_normal([self.num_filters, 1, 1]))
            input = tf.expand_dims(self.X, -1)
            conv = tf.nn.conv1d(input, filter, stride=1, padding='SAME')

            feedforward_input = tf.squeeze(conv, -1)
        else:
            feedforward_input = self.X

        # Weight initializations
        weights = {
            'w1': tf.Variable(tf.random_normal([x_size, self.h1_size]), name='w1'),
            'w2': tf.Variable(tf.random_normal([self.h1_size, self.h2_size]), name='w2'),
            'w3': tf.Variable(tf.random_normal([self.h2_size, self.h3_size]), name='w3'),
            'w4': tf.Variable(tf.random_normal([self.h3_size, self.h4_size]), name='w4'),
            'w5': tf.Variable(tf.random_normal([self.h4_size, self.h5_size]), name='w5'),
            'w6': tf.Variable(tf.random_normal([self.h5_size, self.h6_size]), name='w6')
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([self.h1_size]), name='b1'),
            'b2': tf.Variable(tf.random_normal([self.h2_size]), name='b2'),
            'b3': tf.Variable(tf.random_normal([self.h3_size]), name='b3'),
            'b4': tf.Variable(tf.random_normal([self.h4_size]), name='b4'),
            'b5': tf.Variable(tf.random_normal([self.h5_size]), name='b5'),
            'b6': tf.Variable(tf.random_normal([self.h6_size]), name='b6'),
            'out': tf.Variable(tf.random_normal([y_size]), name='bout')
        }

        # Forward propagation
        h1 = tf.nn.dropout(tf.add(tf.matmul(feedforward_input, weights['w1']), biases['b1']), self.keep_prob)
        h2 = tf.nn.dropout(tf.add(tf.matmul(h1, weights['w2']), biases['b2']), self.keep_prob)
        h3 = tf.nn.dropout(tf.add(tf.matmul(h2, weights['w3']), biases['b3']), self.keep_prob)
        h4 = tf.nn.dropout(tf.add(tf.matmul(h3, weights['w4']), biases['b4']), self.keep_prob)
        h5 = tf.nn.dropout(tf.add(tf.matmul(h4, weights['w5']), biases['b5']), self.keep_prob)
        h6 = tf.nn.dropout(tf.add(tf.matmul(h5, weights['w6']), biases['b6']), self.keep_prob)

        # if h6 is specified
        if self.h6_size:
            weights['out'] = tf.Variable(tf.random_normal([self.h6_size, y_size]), name='wout')
            h6 = tf.nn.tanh(h6)

            yhat = tf.add(tf.matmul(h6, weights['out']), biases['out'])

        # if h5 is specified
        elif self.h5_size:
            weights['out'] = tf.Variable(tf.random_normal([self.h5_size, y_size]), name='wout')
            h5 = tf.nn.sigmoid(h5)

            yhat = tf.add(tf.matmul(h5, weights['out']), biases['out'])

        # if h4 is specified
        elif self.h4_size:
            weights['out'] = tf.Variable(tf.random_normal([self.h4_size, y_size]), name='wout')
            h4 = tf.nn.sigmoid(h4)

            yhat = tf.add(tf.matmul(h4, weights['out']), biases['out'])

        # if h3 is specified
        elif self.h3_size:
            weights['out'] = tf.Variable(tf.random_normal([self.h3_size, y_size]), name='wout')
            h3 = tf.nn.tanh(h3)

            yhat = tf.add(tf.matmul(h3, weights['out']), biases['out'])
        else:
            weights['out'] = tf.Variable(tf.random_normal([self.h2_size, y_size]), name='wout')
            yhat = tf.add(tf.matmul(h2, weights['out']), biases['out'])

        l2_loss = 0.0
        if self.l2_beta:
            l2_loss = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2'])
            if self.h3_size:
                l2_loss += tf.nn.l2_loss(weights['w3'])
            if self.h4_size:
                l2_loss += tf.nn.l2_loss(weights['w4'])
            if self.h5_size:
                l2_loss += tf.nn.l2_loss(weights['w5'])
            if self.h6_size:
                l2_loss += tf.nn.l2_loss(weights['w6'])
            l2_loss = self.l2_beta * l2_loss

        return yhat, l2_loss

    # input and output sizes: x_size, y_size
    def build_model(self, x_size, y_size):
        self.create_placeholders(x_size, y_size)

        # Forward propagation
        self.yhat, l2_loss = self.forward_propagation(x_size, y_size)

        # Backward propagation
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.yhat, targets=self.y) + l2_loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.pred = tf.nn.sigmoid(self.yhat, name='prediction')
