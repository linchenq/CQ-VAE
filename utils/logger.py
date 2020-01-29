import tensorflow as tf
import logging


class Logger(object):
    def __init__(self, log_dir, task_name):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

        """Create a file writter logging to .log"""
        self.logger = logging.getLogger("VAE")
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'{log_dir}model_{task_name}.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.propagate = False

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)

    def log(self, mode, msg):
        if mode == "INF":
            self.logger.info(msg)
        elif mode == "WAR":
            self.logger.warning(msg)
        elif mode == "BUG":
            self.logger.debug(msg)
        else:
            raise NotImplementedError
