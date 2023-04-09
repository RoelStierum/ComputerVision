import tensorflow as tf

model = tf.keras.models.load_model('inference_graph/saved_model/')
model.summary()