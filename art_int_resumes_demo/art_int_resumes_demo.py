"""
DOCSTRING
"""
import csv
import datetime
import itertools
import numpy
import os
import re
import tensorflow
import time

class DataHelpers:
    """
    DOCSTRING
    """
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = numpy.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # shuffle the data at each epoch
            if shuffle:
                shuffle_indices = numpy.random.permutation(numpy.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        source: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_data_and_labels(positive_data_file, negative_data_file):
        """
        Loads MR polarity data from files, splits the data into words, and generates labels.
        
        Returns:
            split sentences and labels
        """
        # load data from files
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # split by words
        x_text = positive_examples + negative_examples
        x_text = [clean_str(sent) for sent in x_text]
        # generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = numpy.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

class Eval:
    """
    DOCSTRING
    """
    def __call__(self):
        # data parameters
        tensorflow.flags.DEFINE_string(
            "positive_data_file",
            "./data/rt-polaritydata/rt-polarity.pos",
            "Data source for the positive data.")
        tensorflow.flags.DEFINE_string(
            "negative_data_file",
            "./data/rt-polaritydata/rt-polarity.neg",
            "Data source for the negative data.")
        # eval parameters
        tensorflow.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
        tensorflow.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
        tensorflow.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
        # misc parameters
        tensorflow.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
        tensorflow.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
        FLAGS = tensorflow.flags.FLAGS
        FLAGS._parse_flags()
        print("\nParameters:")
        for attr, value in sorted(FLAGS.__flags.items()):
            print("{}={}".format(attr.upper(), value))
        print("")
        # NOTE: load your own data here
        if FLAGS.eval_train:
            x_raw, y_test = DataHelpers.load_data_and_labels(
                FLAGS.positive_data_file, FLAGS.negative_data_file)
            y_test = numpy.argmax(y_test, axis=1)
        else:
            x_raw = ["a masterpiece four years in the making", "everything is off."]
            y_test = [1, 0]
        # map data into vocabulary
        vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
        vocab_processor = \
            tensorflow.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_test = numpy.array(list(vocab_processor.transform(x_raw)))
        print("\nEvaluating...\n")
        # evaluation
        checkpoint_file = tensorflow.train.latest_checkpoint(FLAGS.checkpoint_dir)
        graph = tensorflow.Graph()
        with graph.as_default():
            session_conf = tensorflow.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tensorflow.Session(config=session_conf)
            with sess.as_default():
                # load the saved meta graph and restore variables
                saver = tensorflow.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
                # get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                #input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                # tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                # generate batches for one epoch
                batches = DataHelpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
                # collect the predictions here
                all_predictions = list()
                for x_test_batch in batches:
                    batch_predictions = sess.run(
                        predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = numpy.concatenate([all_predictions, batch_predictions])
        # print accuracy if y_test is defined
        if y_test is not None:
            correct_predictions = float(sum(all_predictions == y_test))
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
        # save the evaluation to a csv
        predictions_human_readable = numpy.column_stack((numpy.array(x_raw), all_predictions))
        out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
        print("Saving evaluation to {0}".format(out_path))
        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)

class TextCNN:
    """
    CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
        self,
        sequence_length,
        num_classes,
        vocab_size,
        embedding_size,
        filter_sizes,
        num_filters,
        l2_reg_lambda=0.0):
        # placeholders for input, output and dropout
        self.input_x = tensorflow.placeholder(tensorflow.int32, [None, sequence_length], name="input_x")
        self.input_y = tensorflow.placeholder(tensorflow.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tensorflow.placeholder(tensorflow.float32, name="dropout_keep_prob")
        # keeping track of l2 regularization loss (optional)
        l2_loss = tensorflow.constant(0.0)
        # embedding layer
        with tensorflow.device('/cpu:0'), tensorflow.name_scope("embedding"):
            self.W = tensorflow.Variable(
                tensorflow.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tensorflow.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tensorflow.expand_dims(self.embedded_chars, -1)
        # create a convolution + maxpool layer for each filter size
        pooled_outputs = list()
        for i, filter_size in enumerate(filter_sizes):
            with tensorflow.name_scope("conv-maxpool-%s" % filter_size):
                # convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tensorflow.Variable(tensorflow.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tensorflow.Variable(tensorflow.constant(0.1, shape=[num_filters]), name="b")
                conv = tensorflow.nn.conv2d(
                    self.embedded_chars_expanded, W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # apply nonlinearity
                h = tensorflow.nn.relu(tensorflow.nn.bias_add(conv, b), name="relu")
                # maxpooling over the outputs
                pooled = tensorflow.nn.max_pool(
                    h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tensorflow.concat(pooled_outputs, 3)
        self.h_pool_flat = tensorflow.reshape(self.h_pool, [-1, num_filters_total])
        # add dropout
        with tensorflow.name_scope("dropout"):
            self.h_drop = tensorflow.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        # final (unnormalized) scores and predictions
        with tensorflow.name_scope("output"):
            W = tensorflow.get_variable(
                "W", shape=[num_filters_total, num_classes],
                initializer=tensorflow.contrib.layers.xavier_initializer())
            b = tensorflow.Variable(tensorflow.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tensorflow.nn.l2_loss(W)
            l2_loss += tensorflow.nn.l2_loss(b)
            self.scores = tensorflow.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tensorflow.argmax(self.scores, 1, name="predictions")
        # calculate mean cross-entropy loss
        with tensorflow.name_scope("loss"):
            losses = tensorflow.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tensorflow.reduce_mean(losses) + l2_reg_lambda * l2_loss
        # accuracy
        with tensorflow.name_scope("accuracy"):
            correct_predictions = tensorflow.equal(self.predictions, tensorflow.argmax(self.input_y, 1))
            self.accuracy = tensorflow.reduce_mean(
                tensorflow.cast(correct_predictions, "float"), name="accuracy")

class Train:
    """
    DOCSTRING
    """
    def __init__(self):
        # data loading params
        tensorflow.flags.DEFINE_float(
            "dev_sample_percentage", 0.1,
            "Percentage of the training data to use for validation")
        tensorflow.flags.DEFINE_string(
            "positive_data_file",
            "./data/rt-polaritydata/rt-polarity.pos",
            "Data source for the positive data.")
        tensorflow.flags.DEFINE_string(
            "negative_data_file",
            "./data/rt-polaritydata/rt-polarity.neg",
            "Data source for the negative data.")
        # model hyperparameters
        tensorflow.flags.DEFINE_integer(
            "embedding_dim", 128,
            "Dimensionality of character embedding (default: 128)")
        tensorflow.flags.DEFINE_string(
            "filter_sizes", "3,4,5",
            "Comma-separated filter sizes (default: '3,4,5')")
        tensorflow.flags.DEFINE_integer(
            "num_filters", 128,
            "Number of filters per filter size (default: 128)")
        tensorflow.flags.DEFINE_float(
            "dropout_keep_prob", 0.5,
            "Dropout keep probability (default: 0.5)")
        tensorflow.flags.DEFINE_float(
            "l2_reg_lambda", 0.0,
            "L2 regularization lambda (default: 0.0)")
        # training parameters
        tensorflow.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
        tensorflow.flags.DEFINE_integer(
            "num_epochs", 200,
            "Number of training epochs (default: 200)")
        tensorflow.flags.DEFINE_integer(
            "evaluate_every", 100,
            "Evaluate model on dev set after this many steps (default: 100)")
        tensorflow.flags.DEFINE_integer(
            "checkpoint_every", 100,
            "Save model after this many steps (default: 100)")
        tensorflow.flags.DEFINE_integer(
            "num_checkpoints", 5,
            "Number of checkpoints to store (default: 5)")
        # misc parameters
        tensorflow.flags.DEFINE_boolean(
            "allow_soft_placement", True, "Allow device soft device placement")
        tensorflow.flags.DEFINE_boolean(
            "log_device_placement", False, "Log placement of ops on devices")
        FLAGS = tensorflow.flags.FLAGS
        #FLAGS._parse_flags()
        #print("\nParameters:")
        #for attr, value in sorted(FLAGS.__flags.items()):
        #    print("{}={}".format(attr.upper(), value))
        #print("")

    def dev_step(self, x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: 1.0}
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    def preprocess(self):
        """
        DOCSTRING
        """
        # load data
        print("Loading data...")
        x_text, y = DataHelpers.load_data_and_labels(
            FLAGS.positive_data_file, FLAGS.negative_data_file)
        # build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = \
            tensorflow.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
        x = numpy.array(list(vocab_processor.fit_transform(x_text)))
        # randomly shuffle data
        numpy.random.seed(10)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        # Split train/test set
        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        del x, y, x_shuffled, y_shuffled
        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        return x_train, y_train, vocab_processor, x_dev, y_dev

    def train(self, x_train, y_train, vocab_processor, x_dev, y_dev):
        """
        DOCSTRING
        """
        with tensorflow.Graph().as_default():
            session_conf = tensorflow.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tensorflow.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
                # define Training procedure
                global_step = tensorflow.Variable(0, name="global_step", trainable=False)
                optimizer = tensorflow.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                # keep track of gradient values and sparsity (optional)
                grad_summaries = list()
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tensorflow.summary.histogram(
                            "{}/grad/hist".format(v.name), g)
                        sparsity_summary = tensorflow.summary.scalar(
                            "{}/grad/sparsity".format(v.name), tensorflow.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tensorflow.summary.merge(grad_summaries)
                # output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))
                # summaries for loss and accuracy
                loss_summary = tensorflow.summary.scalar("loss", cnn.loss)
                acc_summary = tensorflow.summary.scalar("accuracy", cnn.accuracy)
                # train summaries
                train_summary_op = tensorflow.summary.merge(
                    [loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tensorflow.summary.FileWriter(train_summary_dir, sess.graph)
                # dev summaries
                dev_summary_op = tensorflow.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tensorflow.summary.FileWriter(dev_summary_dir, sess.graph)
                # checkpoint directory
                # Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tensorflow.train.Saver(tensorflow.global_variables(), max_to_keep=FLAGS.num_checkpoints)
                # write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))
                # initialize all variables
                sess.run(tensorflow.global_variables_initializer())
                # generate batches
                batches = DataHelpers.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                # training loop
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    self.train_step(x_batch, y_batch)
                    current_step = tensorflow.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        self.dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

    def train_step(self, x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tensorflow.app.run()
