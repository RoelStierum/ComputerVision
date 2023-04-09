import os
import pathlib
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import csv

# Path to the directory containing validation images
IMAGE_DIR = 'images/val/'

# Path to the frozen inference graph
MODEL_DIR = 'inference_graph/saved_model/'

# Path to the label map
LABELMAP_PATH = 'labelmap.pbtxt'

# Path to the directory where you want to save the output images
OUTPUT_DIR = 'output/'

img_height = 512
img_width = 512

# Load the saved model
detection_model = tf.saved_model.load(MODEL_DIR)

# Load the label map
category_index = label_map_util.create_category_index_from_labelmap(LABELMAP_PATH, use_display_name=True)

# Get the list of validation images
validation_image_paths = list(pathlib.Path(IMAGE_DIR).glob('*.jpg'))
validation_images = []
for image_path in validation_image_paths:
    image = load_img(image_path, target_size=(img_height, img_width))
    image_array = img_to_array(image)
    validation_images.append(image_array)

# Convert the validation images to numpy arrays
validation_images = np.array(validation_images, dtype=object)
validation_images = validation_images / 255.0

# Get the validation image filenames
validation_image_filenames = [os.path.basename(image_path) for image_path in validation_image_paths]

# Read the label CSV file
with open('images/val_labels.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    labels = []
    for row in csv_reader:
        labels.append(row)

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run object detection on the validation images and save the output images
num_correct = 0
num_total = 0
for i in range(len(validation_images)):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(validation_images[i], axis=0)

    # Run inference
    detections = detection_model(image_np_expanded)

    # Get the labels for the current image
    current_labels = [label for label in labels if label['filename'] == validation_image_filenames[i]]

    # Visualize the results of object detection on the validation image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        validation_images[i],
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.3,
        line_thickness=2)

    # Calculate accuracy
    num_labels = len(current_labels)
    num_total += num_labels
    num_detected = len(detections['detection_boxes'][0])
    num_correct += min(num_detected, num_labels)

    validation_images = (validation_images * 255).astype(np.uint8)
    validation_images[i] = np.array(Image.fromarray(validation_images[i]).resize((image.width, image.height)))

    # Save the output image
    output_image_path = os.path.join(OUTPUT_DIR, validation_image_filenames[i])
    Image.fromarray(validation_images[i]).save(output_image_path)

# Calculate and print accuracy
accuracy = num_correct / num_total

print('num correct')
print(num_correct)
print("num total")
print(num_total)
print(f'{accuracy:.5%}')