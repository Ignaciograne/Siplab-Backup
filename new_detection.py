import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from picamera2 import Picamera2
import utils  # Asegúrate de que este módulo esté disponible
import datetime

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, output_filename: str, fps: int) -> None:
    """Run object detection and record video with detections and timestamp.

    Args:
        model: Path to the TFLite object detection model.
        camera_id: Camera ID (no se usa con Picamera2, pero lo mantenemos por compatibilidad).
        width: Frame width.
        height: Frame height.
        num_threads: Número de hilos para el modelo.
        enable_edgetpu: Si usar EdgeTPU.
        output_filename: Nombre del archivo de video de salida.
        fps: Frames por segundo para el video de salida.
    """

    # Variables para calcular FPS
    counter, fps_display = 0, 0
    start_time = time.time()

    # Inicializa Picamera2
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (width, height)})
    picam2.configure(video_config)
    picam2.start()

    # Configura el VideoWriter para guardar el video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Parámetros de visualización
    row_size = 20  # píxeles
    left_margin = 24  # píxeles
    text_color = (0, 0, 255)  # rojo
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Inicializa el modelo de detección de objetos
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    print("Iniciando detección y grabación de video...")

    try:
        while True:
            # Captura el cuadro del video
            frame = picam2.capture_array()

            # Incrementa el contador de cuadros
            counter += 1

            # Convierte el marco de RGB a BGR para OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Ejecuta la detección de objetos
            rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = vision.TensorImage.create_from_array(rgb_image)
            detection_result = detector.detect(input_tensor)

            # Dibuja las detecciones en el cuadro
            frame_bgr, count = utils.visualize(frame_bgr, detection_result)

            # Calcula el FPS
            if counter % fps_avg_frame_count == 0:
                end_time = time.time()
                fps_display = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()

            # Agrega la información de FPS y conteo al cuadro
            fps_text = f'FPS = {fps_display:.1f} | Objetos detectados = {count}'
            text_location = (left_margin, row_size)
            cv2.putText(frame_bgr, fps_text, text_location, cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, text_color, font_thickness)

            # Agrega la marca de tiempo al cuadro
            local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_bgr, local_time, (10, frame_bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Muestra el cuadro en una ventana (puedes omitir esto si no lo necesitas)
            cv2.imshow('Detección de Objetos', frame_bgr)

            # Escribe el cuadro en el archivo de video
            out.write(frame_bgr)

            # Maneja la salida si se presiona la tecla ESC
            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        # Permite detener la grabación con Ctrl+C
        print("\nDeteniendo la grabación...")

    finally:
        # Libera los recursos
        picam2.stop()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video guardado como {output_filename}")

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Ruta del modelo de detección de objetos.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--cameraId', 
        help='ID de la cámara (no se usa con Picamera2).', 
        required=False, 
        type=int, 
        default=0)
    parser.add_argument(
        '--frameWidth',
        help='Ancho del cuadro a capturar de la cámara.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Altura del cuadro a capturar de la cámara.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Número de hilos de CPU para ejecutar el modelo.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Si ejecutar el modelo en EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--output',
        help='Nombre del archivo de video de salida.',
        required=False,
        default='video_detectado.mp4')
    parser.add_argument(
        '--fps',
        help='Frames por segundo para el video de salida.',
        required=False,
        type=int,
        default=10)
    args = parser.parse_args()

    run(args.model, args.cameraId, args.frameWidth, args.frameHeight,
        args.numThreads, args.enableEdgeTPU, args.output, args.fps)

if __name__ == '__main__':
    main()
