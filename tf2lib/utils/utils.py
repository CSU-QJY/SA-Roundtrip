import tensorflow as tf


class Checkpoint:
    """Enhanced "tf.train.Checkpoint"."""

    def __init__(self,
                 checkpoint_kwargs,  # for "tf.train.Checkpoint"
                 directory,  # for "tf.train.CheckpointManager"
                 max_to_keep=5,
                 keep_checkpoint_every_n_hours=None):
        self.checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep,
                                                  keep_checkpoint_every_n_hours)

    def restore(self, save_path=None):
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.checkpoint.restore(save_path)

    def save(self, file_prefix_or_checkpoint_number=None, session=None):
        if isinstance(file_prefix_or_checkpoint_number, str):
            return self.checkpoint.save(file_prefix_or_checkpoint_number, session=session)
        else:
            return self.manager.save(checkpoint_number=file_prefix_or_checkpoint_number)

    def __getattr__(self, attr):
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return getattr(self.manager, attr)
        else:
            self.__getattribute__(attr)  # this will raise an exception


def summary(name_data_dict,
            step=None,
            types=None,
            historgram_buckets=None,
            scope='summary'):
    if types is None:
        types = ['mean', 'std', 'max', 'min', 'sparsity', 'histogram', 'image']

    def _summary(name, data):
        if scope == 'Images':
            tf.summary.image(name, tf.expand_dims(data, axis=0), step=step)
        elif scope == 'Distribution':
            tf.summary.histogram(name, data, step=step)
        else:
            tf.summary.scalar(name, data, step=step)
        # else:
        #     if 'mean' in types:
        #         tf.summary.scalar(name + '-mean', tf.math.reduce_mean(data), step=step)
        #     if 'std' in types:
        #         tf.summary.scalar(name + '-std', tf.math.reduce_std(data), step=step)
        #     if 'max' in types:
        #         tf.summary.scalar(name + '-max', tf.math.reduce_max(data), step=step)
        #     if 'min' in types:
        #         tf.summary.scalar(name + '-min', tf.math.reduce_min(data), step=step)
        #     if 'sparsity' in types:
        #         tf.summary.scalar(name + '-sparsity', tf.math.zero_fraction(data), step=step)
        #     if 'histogram' in types:
        #         tf.summary.histogram(name, data, step=step, buckets=historgram_buckets)
        #     if 'image' in types:
        #         tf.summary.image('Generated Images', tf.expand_dims(data, axis=0), step=step)

    with tf.name_scope(scope):
        for name, data in name_data_dict.items():
            _summary(name, data)
