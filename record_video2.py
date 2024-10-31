import time
import cv2
from picamera2 import Picamera2

# Inicializa la cámara
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(video_config)

# Parámetros de grabación
fps = 10  # Ajustar a un fps más bajo para hacer el video más lento
output_filename = "video_con_hora.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (640, 480))

# Comienza la grabación
picam2.start()

print("Grabando video y hora indefinidamente...")

try:
    while True:
        # Captura el marco del video
        frame = picam2.capture_array()

        # Obtiene la hora local en tiempo real (UTC-6)
        local_time_epoch = time.time() - 6 * 3600  # Ajustar a UTC-6
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(local_time_epoch))

        # Convierte el marco a BGR para OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Agrega el texto de la hora al marco del video
        cv2.putText(frame_bgr, local_time, (10, frame_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Escribe el cuadro en el archivo de video
        out.write(frame_bgr)

        # Ajustar el tiempo de espera para hacer el video más lento
        time.sleep((1 / fps)*0.8)  # Se ajusta para que se mantenga el fps

except KeyboardInterrupt:
    # Se puede usar Ctrl+C para detener la grabación
    print("Deteniendo la grabación...")

finally:
    # Libera los recursos
    picam2.stop()  # Detiene la grabación
    out.release()  # Libera el VideoWriter

print(f"Video guardado como {output_filename}")

