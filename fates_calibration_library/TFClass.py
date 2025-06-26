"""TensorFlow Emulator Class"""
import os
import tensorflow as tf

class TFEmulator:
    def __init__(self, model_dir, pft, variable):
        path = os.path.join(model_dir, f"{pft}_{variable}")
        self.loaded = tf.saved_model.load(path)
        self.predict_fn = self.loaded.signatures["serving_default"]

    def __call__(self, X):
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float64)
        output = self.predict_fn(X=X_tensor)
        return output["mean"], output["variance"]