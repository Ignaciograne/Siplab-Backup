import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

import os
import glob
import csv  # Importamos csv para manejar la escritura del archivo CSV

def run(model: str, image_path:str, width: int, height: int, num_threads: int, enable_edgetpu: bool, output_csv: str) -> None:
  images = glob.glob(os.path.join(image_path, '*jpg'))

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=100, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
  
  # Preparamos el archivo CSV para guardar los resultados
  with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['Image Path', 'Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Continuously capture images from the camera and run inference
    for img_path in images:
      image = cv2.imread(img_path)
      if image is None:
          continue

      rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      input_tensor = vision.TensorImage.create_from_array(rgb_image)
      detection_result = detector.detect(input_tensor)

      image, count = utils.visualize(image, detection_result)
      print("For image: ", img_path, " founded ", count, " people")
      
      # Guardamos los resultados en el CSV
      writer.writerow({'Image Path': img_path, 'Count': count})

      # Guardamos las imágenes
      # Show the FPS
      fps_text = 'Counted objets = {counter}'.format(counter=count)
      text_location = (left_margin, row_size)
      cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

      base_name = os.path.basename(img_path)
      cv2.imwrite(f"./processed_{base_name}", image)
      print("Saved ", base_name, " as processed_",base_name)

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model', help='Path of the object detection model.', required=False, default='efficientdet_lite0.tflite')
  parser.add_argument('--imagePath', help='Path to the image or directory of images.', required=True)
  parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, type=int, default=640)
  parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, type=int, default=480)
  parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', required=False, type=int, default=4)
  parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true', required=False, default=False)
  parser.add_argument('--outputCsv', help='Output CSV file to save the results.', required=False, default='detection_results.csv')
  args = parser.parse_args()

  run(args.model, args.imagePath, args.frameWidth, args.frameHeight, int(args.numThreads), bool(args.enableEdgeTPU), args.outputCsv)

if __name__ == '__main__':
  main()

