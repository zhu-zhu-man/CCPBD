import numpy as np

A = np.load("G:\images_path.npy")
for i in range(len(A)):
    print(A[i])


    @keras_export('keras.metrics.Kappa')
    class Kappa(keras.metrics.Metric):
        def __init__(self,
                     thresholds=None,
                     top_k=None,
                     class_id=None,
                     name="Kappa",
                     dtype=None):
            super(Kappa, self).__init__(name=name, dtype=dtype)
            self.init_thresholds = thresholds
            self.top_k = top_k
            self.class_id = class_id

            default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
            self.thresholds = metrics_utils.parse_init_thresholds(
                thresholds, default_threshold=default_threshold)
            self.true_negatives = self.add_weight(
                'true_negatives',
                shape=(len(self.thresholds),),
                initializer=init_ops.zeros_initializer)
            self.true_positives = self.add_weight(
                'true_positives',
                shape=(len(self.thresholds),),
                initializer=init_ops.zeros_initializer)
            self.false_negatives = self.add_weight(
                'false_negatives',
                shape=(len(self.thresholds),),
                initializer=init_ops.zeros_initializer)
            self.false_positives = self.add_weight(
                'false_positives',
                shape=(len(self.thresholds),),
                initializer=init_ops.zeros_initializer)

        def update_state(self, y_true, y_pred, sample_weight=None):
            return metrics_utils.update_confusion_matrix_variables(
                {
                    metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                    metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
                },
                y_true,
                y_pred,
                thresholds=self.thresholds,
                top_k=self.top_k,
                class_id=self.class_id,
                sample_weight=sample_weight)

        def result(self):
            Acc = math_ops.div_no_nan(self.true_positives + self.true_negatives,
                                      self.true_negatives + self.true_positives + self.false_negatives + self.false_positives)
            Pe = math_ops.div_no_nan(
                (self.true_positives + self.false_positives) * (self.true_positives + self.false_negatives) + (
                            self.true_negatives + self.false_negatives) * (self.true_negatives + self.false_positives),
                (self.true_negatives + self.true_positives + self.false_negatives + self.false_positives) * (
                            self.true_negatives + self.true_positives + self.false_negatives + self.false_positives))
            # Kappa
            result = math_ops.div_no_nan(Acc - Pe,
                                         1 - Pe)
            return result[0] if len(self.thresholds) == 1 else result

        def reset_state(self):
            num_thresholds = len(to_list(self.thresholds))
            K.batch_set_value(
                [(v, np.zeros((num_thresholds,))) for v in self.variables])

        def get_config(self):
            config = {
                'thresholds': self.init_thresholds,
                'top_k': self.top_k,
                'class_id': self.class_id
            }
            base_config = super(Kappa, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))






    @keras_export('keras.metrics.F1Score')
    class F1Score(keras.metrics.Metric):
        def __init__(self,
                     thresholds=None,
                     top_k=None,
                     class_id=None,
                     name="F1Score",
                     dtype=None):
            super(F1Score, self).__init__(name=name, dtype=dtype)
            self.init_thresholds = thresholds
            self.top_k = top_k
            self.class_id = class_id

            default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
            self.thresholds = metrics_utils.parse_init_thresholds(
                thresholds, default_threshold=default_threshold)
            '''
             self.true_negatives = self.add_weight(
                'true_negatives',
                shape=(len(self.thresholds),),
                initializer=init_ops.zeros_initializer)         
            '''

            self.true_positives = self.add_weight(
                'true_positives',
                shape=(len(self.thresholds),),
                initializer=init_ops.zeros_initializer)
            self.false_negatives = self.add_weight(
                'false_negatives',
                shape=(len(self.thresholds),),
                initializer=init_ops.zeros_initializer)
            self.false_positives = self.add_weight(
                'false_positives',
                shape=(len(self.thresholds),),
                initializer=init_ops.zeros_initializer)

        def update_state(self, y_true, y_pred, sample_weight=None):
            return metrics_utils.update_confusion_matrix_variables(
                {
                    # metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                    metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
                },
                y_true,
                y_pred,
                thresholds=self.thresholds,
                top_k=self.top_k,
                class_id=self.class_id,
                sample_weight=sample_weight)

        def result(self):
            Recall = math_ops.div_no_nan(self.true_positives,
                                         self.true_positives + self.false_negatives)
            Precision = math_ops.div_no_nan(self.true_positives,
                                            self.true_positives + self.false_positives)

            # F1Score
            result = math_ops.div_no_nan(2 * Recall * Precision,
                                         Recall + Precision)
            return result[0] if len(self.thresholds) == 1 else result

        def reset_state(self):
            num_thresholds = len(to_list(self.thresholds))
            K.batch_set_value(
                [(v, np.zeros((num_thresholds,))) for v in self.variables])

        def get_config(self):
            config = {
                'thresholds': self.init_thresholds,
                'top_k': self.top_k,
                'class_id': self.class_id
            }
            base_config = super(F1Score, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'accuracy',
        patience = 3,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath = '/kaggle/working/my_model.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2)
]

