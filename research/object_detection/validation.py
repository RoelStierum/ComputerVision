import tensorflow as tf
import pandas as pd
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Load the label map
label_map_path = 'labelmap.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the model
model = tf.saved_model.load("inference_graph/saved_model")

# Load the validation dataset from csv
validation_csv_path = 'images/val_labels.csv'
validation_dataset = pd.read_csv(validation_csv_path)

# Create a summary writer for TensorBoard
log_dir = 'training/train'
summary_writer = tf.summary.create_file_writer(log_dir)

# Initialize the accuracy metric
m = tf.keras.metrics.Accuracy()

# Evaluate the model on the validation dataset
for idx, row in validation_dataset.iterrows():
    image_path = 'images/val/' + row['filename']
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    detections = model(tf.cast(image, tf.uint8)[tf.newaxis, ...])
    detections['detection_scores'] = tf.reshape(detections['detection_scores'], (1, -1))
    print(detections['detection_scores'].shape)
    num_detections = tf.cast(detections['num_detections'][0], tf.int32)
    detections = tf.reshape(detections['detection_scores'], (1, -1, len(categories)))
    if tf.size(detections) == 0:  # check if detections is empty
        continue
    print(detections['detection_scores'].shape)
    scores = detections['detection_scores'][0, :num_detections].numpy()  # get all detection scores
    classes = detections[:, :, 5].astype(int)  # get all detection classes
    boxes = detections[:, :, :4] # get all detection boxes
    label = row['class']
    label_idx = [category['id'] for category in categories if category['name'] == label][0]
    one_hot = tf.one_hot(classes, len(categories), dtype=tf.int32)
    one_hot = tf.cast(one_hot, tf.float32) * label_idx
    y_pred = tf.reduce_max(scores * one_hot, axis=-1)
    y_true = tf.cast(tf.reduce_max(one_hot, axis=-1), tf.float32)
    m.update_state(y_true, y_pred)
    with summary_writer.as_default():
        tf.summary.scalar('accuracy', m.result().numpy(), step=idx+1)

# Close the summary writer
summary_writer.close()