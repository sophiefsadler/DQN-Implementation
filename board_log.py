import tensorflow as tf

class Tensorboard(object):

    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def log(self, label, value, count):
        summary = tf.Summary()
        summary.value.add(tag=label, simple_value=value)
        self.writer.add_summary(summary, global_step=count)
        self.writer.flush()

    def close(self):
        self.writer.close()
