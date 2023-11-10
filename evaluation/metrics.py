import tensorflow as tf
import seaborn as sn

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, classnum, name="confusion_matrix"):
        super(ConfusionMatrix, self).__init__(name=name)
        self.classnum = classnum
        self.confusionmatrix = self.add_weight(shape=(classnum, classnum), name='cm', initializer='zeros')

    def update_state(self, true, prediction):
        self.confusionmatrix.assign_add(
            tf.math.confusion_matrix(true, prediction, dtype=tf.float32, num_classes=self.classnum))

    def result(self):
        return self.confusionmatrix

    def reset_state(self):
        self.confusionmatrix.assign(tf.zeros([self.classnum, self.classnum]))

    def metric(self):
        cm = self.confusionmatrix
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (recall + precision)
        return precision, recall, f1


def pltmatrix(cm):
  hmap = sn.heatmap(cm,annot=True)
  hmap.set_xlabel('Predict')
  hmap.set_ylabel('True')