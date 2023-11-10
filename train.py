import gin
import tensorflow as tf
import logging
import wandb

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, total_steps, log_interval, lr, activation, check_path):
        # Summary Writer
        # ....

        # Checkpoint Manager

        # Loss objective
        self.lr = lr

        if activation == 'sigmoid':
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
            self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
        else:
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.weight = [0, 0]
        self.weight[0] = ds_info['len']['train'] / ds_info['num_0']['train']
        self.weight[1] = ds_info['len']['train'] / ds_info['num_1']['train']
        self.total_steps = total_steps
        self.log_interval = log_interval
        # self.ckpt_interval = ckpt_interval

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.check_path = check_path
        self.checkmanager = tf.train.CheckpointManager(self.checkpoint, directory=self.check_path, max_to_keep=10)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
            weight = tf.gather(self.weight, labels)
            loss = tf.math.reduce_mean(tf.math.multiply(weight, loss))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))
                # weight and biases
                wandb.log({'step': step,
                           'loss': self.train_loss.result(),
                           'acc': self.train_accuracy.result() * 100,
                           'val_loss': self.val_loss.result(),
                           'val_acc': self.val_accuracy.result() * 100})

                # Write summary to tensorboard
                # ...

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            # if step % self.ckpt_interval == 0:
            #     logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
            #     # Save checkpoint
            #     # ...
            if self.val_accuracy.result() * 100 > 87 and step % self.log_interval == 0:
                logging.info(f'Saving checkpoint to {self.check_path}.')
                # Save checkpoint
                saved_path = self.checkmanager.save()
                print("save checkpoint for step {} at {}".format(int(step), saved_path))

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                # ...
                return self.val_accuracy.result().numpy()
