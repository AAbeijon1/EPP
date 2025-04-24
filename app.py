from flask import Flask, render_template, request, jsonify, send_from_directory, Response, url_for
import os
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import json
import logging
import traceback
from datetime import datetime
import shutil
from PIL import Image, ImageDraw, ImageStat
import base64
import io
import threading
import time
import sys
import math
from collections import defaultdict
import colorsys
import re
import colorsys
from datetime import datetime, timedelta
import math
from collections import defaultdict

# Configurar logging
logging.basicConfig(
  level=logging.DEBUG,  # Cambiar a DEBUG para más información
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  handlers=[
      logging.FileHandler("app.log"),
      logging.StreamHandler()
  ]
)  # Falta este paréntesis de cierre
logger = logging.getLogger(__name__)



def extract_dominant_color(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"No se pudo cargar la imagen: {image_path}")
            return [128, 128, 128]  # Gris por defecto
            
        # Redimensionar para acelerar el procesamiento
        img_resized = cv2.resize(img, (50, 50))
        # Convertir a RGB (desde BGR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Reshapear para que sea una lista de píxeles
        pixels = img_rgb.reshape(-1, 3)
        
        # Agrupar colores similares (k-means)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3  # Número de clusters (colores dominantes)
        _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Encontrar el cluster más grande
        counts = np.bincount(labels.flatten())
        dominant_color = centers[np.argmax(counts)]
        
        return dominant_color.astype(int).tolist()
    except Exception as e:
        logger.error(f"Error al extraer color dominante: {e}")
        return [128, 128, 128]  # Gris por defecto


# Función para registrar errores
def log_error(message, exception=None):
  logger.error(message)
  if exception:
      logger.error(traceback.format_exc())

# Configuración de la aplicación Flask
app = Flask(__name__)

# Definir rutas absolutas para todos los directorios
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(BASE_DIR, 'models')
app.config['DETECTION_FOLDER'] = os.path.join(BASE_DIR, 'detections')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['STATIC_FOLDER'] = os.path.join(BASE_DIR, 'static')
app.config['TRAJECTORY_FOLDER'] = os.path.join(BASE_DIR, 'trajectories')  # Nueva carpeta para datos de trayectorias

# Aumentar el límite de tamaño de archivo
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Crear directorios necesarios con rutas absolutas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'videos'), exist_ok=True)
os.makedirs(app.config['TRAJECTORY_FOLDER'], exist_ok=True)  # Crear directorio de trayectorias

# Asegurar que los directorios tengan permisos adecuados
try:
    # En sistemas Unix/Linux
    if sys.platform != 'win32':
        os.chmod(app.config['UPLOAD_FOLDER'], 0o755)
        os.chmod(app.config['MODEL_FOLDER'], 0o755)
        os.chmod(app.config['DETECTION_FOLDER'], 0o755)
        os.chmod(app.config['OUTPUT_FOLDER'], 0o755)
        os.chmod(app.config['STATIC_FOLDER'], 0o755)
        os.chmod(app.config['TRAJECTORY_FOLDER'], 0o755)
except Exception as e:
    log_error(f"No se pudieron establecer permisos en los directorios", e)

# Crear imagen de marcador de posición si no existe
placeholder_path = os.path.join(app.config['STATIC_FOLDER'], 'placeholder.jpg')
if not os.path.exists(placeholder_path):
  try:
      img = Image.new('RGB', (400, 300), color=(240, 240, 240))
      d = ImageDraw.Draw(img)
      d.rectangle([0, 0, 400, 300], outline=(200, 200, 200), width=2)
      d.text((150, 140), "Imagen no disponible", fill=(100, 100, 100))
      img.save(placeholder_path)
      logger.info(f"Created placeholder image at {placeholder_path}")
  except Exception as e:
      log_error(f"Failed to create placeholder image", e)
      # Crear una imagen en blanco como fallback
      try:
          blank_img = Image.new('RGB', (400, 300), color=(240, 240, 240))
          blank_img.save(placeholder_path)
          logger.info(f"Created blank placeholder image at {placeholder_path}")
      except Exception as e2:
          log_error(f"Failed to create blank placeholder image", e2)

# Crear SVG de marcador de posición si no existe
placeholder_svg_path = os.path.join(app.config['STATIC_FOLDER'], 'placeholder.svg')
if not os.path.exists(placeholder_svg_path):
    try:
        svg_content = '''<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="300" fill="#f0f0f0" stroke="#c8c8c8" stroke-width="2"/>
  <text x="200" y="150" font-family="Arial" font-size="16" text-anchor="middle" fill="#646464">Imagen no disponible</text>
</svg>'''
        with open(placeholder_svg_path, 'w') as f:
            f.write(svg_content)
        logger.info(f"Created placeholder SVG at {placeholder_svg_path}")
    except Exception as e:
        log_error(f"Failed to create placeholder SVG", e)

# Almacenamiento en memoria para detecciones
detections_db = []

# Almacenamiento para datos de trayectorias
trajectory_data = {}

# Estado de procesamiento
processing_status = {
  'is_processing': False,
  'progress': 0,
  'total_frames': 0,
  'current_frame': 0,
  'status': 'idle',
  'error': None,
  'output_video': None,
  'detections': [],
  'trajectory_data': {}  # Nuevo campo para datos de trayectoria
}

# Función para copiar archivos a la carpeta estática
def copy_to_static(source_path, file_type='images'):
  try:
      # Generar un nombre único para el archivo
      filename = f"{str(uuid.uuid4())}{os.path.splitext(source_path)[1]}"
      static_path = os.path.join(app.config['STATIC_FOLDER'], file_type, filename)
      
      # Copiar el archivo
      shutil.copy2(source_path, static_path)
      logger.info(f"Copied {source_path} to {static_path}")
      
      # Devolver la ruta relativa para usar en HTML
      return f"/static/{file_type}/{filename}"
  except Exception as e:
      log_error(f"Error copying file to static folder: {source_path}", e)
      return None

# Función para convertir imagen a base64
def image_to_base64(image_path):
  try:
      with open(image_path, "rb") as img_file:
          return base64.b64encode(img_file.read()).decode('utf-8')
  except Exception as e:
      log_error(f"Error converting image to base64: {image_path}", e)
      return None

# Añadir esta función para guardar el video con un nombre más predecible
def save_processed_video(original_path, output_folder):
  try:
      # Generar un nombre basado en la fecha y hora actual
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      video_filename = f"processed_video_{timestamp}.mp4"
      output_path = os.path.join(output_folder, video_filename)
      
      # Asegurarse de que el directorio existe
      os.makedirs(output_folder, exist_ok=True)
      
      # Copiar el video original al nuevo destino
      shutil.copy2(original_path, output_path)
      logger.info(f"Video guardado con nombre predecible: {output_path}")
      
      return output_path, video_filename
  except Exception as e:
      log_error(f"Error al guardar el video procesado con nombre predecible", e)
      return original_path, os.path.basename(original_path)

# Añadir esta función para copiar y renombrar el video procesado
def copy_and_rename_output_video(output_path):
    try:
        # Generar un nombre más simple y reconocible
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        simple_name = f"processed_video_{timestamp}.mp4"
        simple_path = os.path.join(app.config['OUTPUT_FOLDER'], simple_name)
        
        # Copiar el archivo con el nuevo nombre
        shutil.copy2(output_path, simple_path)
        logger.info(f"Video copiado y renombrado: {simple_path}")
        
        # Copiar también a la carpeta estática
        static_path = os.path.join(app.config['STATIC_FOLDER'], 'videos', simple_name)
        shutil.copy2(output_path, static_path)
        logger.info(f"Video copiado a carpeta estática: {static_path}")
        
        return simple_name
    except Exception as e:
        log_error(f"Error al copiar y renombrar video", e)
        return None

# Función para guardar datos de trayectoria
def save_trajectory_data(track_id, frame_number, x, y, class_name, confidence, image_path, frame_width, frame_height):
    """
    Guarda los datos de trayectoria de un objeto detectado.
    
    Args:
        track_id: ID único de seguimiento del objeto
        frame_number: Número del frame actual
        x, y: Coordenadas del centro del objeto
        class_name: Nombre de la clase del objeto
        confidence: Confianza de la detección
        image_path: Ruta a la imagen de la detección
        frame_width, frame_height: Dimensiones del frame para normalizar coordenadas
    """
    global trajectory_data
    
    # Normalizar coordenadas (0-1) para independencia de resolución
    x_normalized = x / frame_width
    y_normalized = y / frame_height
    
    if track_id not in trajectory_data:
        trajectory_data[track_id] = {
            'class': class_name,
            'timestamps': [],
            'positions': [],
            'frames': [],
            'filenames': [],
            'confidences': [],
            'normalized_positions': []
        }
    
    # Añadir datos de trayectoria
    trajectory_data[track_id]['timestamps'].append(datetime.now().timestamp())
    trajectory_data[track_id]['positions'].append({'x': x, 'y': y})
    trajectory_data[track_id]['frames'].append(frame_number)
    trajectory_data[track_id]['filenames'].append(image_path)
    trajectory_data[track_id]['confidences'].append(confidence)
    trajectory_data[track_id]['normalized_positions'].append({'x': x_normalized, 'y': y_normalized})
    
    # Actualizar el estado de procesamiento con los datos de trayectoria
    processing_status['trajectory_data'] = trajectory_data
    
    logger.info(f"Saved trajectory data for track_id {track_id}, frame {frame_number}")

# Añadir encabezados CORS a todas las respuestas
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    return response

@app.route('/')
def index():
  return render_template('index.html')

# Añadir una ruta para verificar la conexión al servidor
@app.route('/ping')
def ping():
    """
    Endpoint simple para verificar que el servidor está funcionando.
    """
    return jsonify({
        'success': True,
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat()
    })

# Mejorar el manejo de errores en las rutas de carga de archivos
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    # Manejar solicitudes OPTIONS para CORS
    if request.method == 'OPTIONS':
        return jsonify({'success': True})
        
    try:
        logger.info(f"Recibida solicitud de carga de archivo. Método: {request.method}")
        logger.info(f"Encabezados: {request.headers}")
        
        # Verificar si hay datos en la solicitud
        if not request.content_length:
            logger.warning("No se recibieron datos en la solicitud")
            return jsonify({'success': False, 'error': 'No data received in request'})
        
        # Verificar si el tamaño del archivo excede el límite
        if request.content_length > app.config['MAX_CONTENT_LENGTH']:
            logger.warning(f"Tamaño de archivo excede el límite: {request.content_length} bytes")
            return jsonify({
                'success': False, 
                'error': f'File size exceeds limit of {app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)} MB'
            })
        
        if 'video' not in request.files:
            logger.warning("No se encontró 'video' en request.files")
            logger.info(f"Claves disponibles en request.files: {list(request.files.keys())}")
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        file = request.files['video']
        logger.info(f"Archivo recibido: {file.filename}, tipo: {file.content_type}, tamaño: {request.content_length} bytes")
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return jsonify({'success': False, 'error': 'File must be a video (mp4, avi, mov)'})
        
        # Generar un nombre único para el archivo
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Guardar el archivo
        file.save(filepath)
        logger.info(f"Video uploaded: {filepath}")
        
        # Verificar que el archivo se guardó correctamente
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Failed to save file'})
        
        file_size = os.path.getsize(filepath)
        logger.info(f"Archivo guardado correctamente: {filepath}, tamaño: {file_size} bytes")
        
        return jsonify({
            'success': True,
            'video_path': filepath,
            'message': 'Video uploaded successfully',
            'file_size': file_size
        })
        
    except Exception as e:
        log_error("Error in upload_file", e)
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})

@app.route('/upload_model', methods=['POST', 'OPTIONS'])
def upload_model():
    # Manejar solicitudes OPTIONS para CORS
    if request.method == 'OPTIONS':
        return jsonify({'success': True})
        
    try:
        logger.info(f"Recibida solicitud de carga de modelo. Método: {request.method}")
        logger.info(f"Encabezados: {request.headers}")
        
        # Verificar si hay datos en la solicitud
        if not request.content_length:
            logger.warning("No se recibieron datos en la solicitud")
            return jsonify({'success': False, 'error': 'No data received in request'})
        
        # Verificar si el tamaño del archivo excede el límite
        if request.content_length > app.config['MAX_CONTENT_LENGTH']:
            logger.warning(f"Tamaño de archivo excede el límite: {request.content_length} bytes")
            return jsonify({
                'success': False, 
                'error': f'File size exceeds limit of {app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)} MB'
            })
        
        if 'model' not in request.files:
            logger.warning("No se encontró 'model' en request.files")
            logger.info(f"Claves disponibles en request.files: {list(request.files.keys())}")
            return jsonify({'success': False, 'error': 'No model file provided'})
        
        file = request.files['model']
        logger.info(f"Archivo de modelo recibido: {file.filename}, tipo: {file.content_type}, tamaño: {request.content_length} bytes")
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.lower().endswith('.pt'):
            return jsonify({'success': False, 'error': 'File must be a YOLOv8 model (.pt)'})
        
        # Generar un nombre único para el archivo
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
        
        # Guardar el archivo
        file.save(filepath)
        logger.info(f"Model uploaded: {filepath}")
        
        # Verificar que el archivo se guardó correctamente
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Failed to save model file'})
        
        file_size = os.path.getsize(filepath)
        logger.info(f"Archivo de modelo guardado correctamente: {filepath}, tamaño: {file_size} bytes")
        
        return jsonify({
            'success': True,
            'model_path': filepath,
            'message': 'Model uploaded successfully',
            'file_size': file_size
        })
        
    except Exception as e:
        log_error("Error in upload_model", e)
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})




# Nueva ruta para el módulo forense
@app.route('/forensic_search')
def forensic_search():
    try:
        # Obtener parámetros de búsqueda
        color_hex = request.args.get('color', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        size_min = request.args.get('size_min', '')
        size_max = request.args.get('size_max', '')
        time_range = request.args.get('time_range', '')  # Formato HH:MM-HH:MM
        
        # Convertir color hexadecimal a RGB
        color_rgb = None
        if color_hex and color_hex.startswith('#') and len(color_hex) == 7:
            color_rgb = [
                int(color_hex[1:3], 16),
                int(color_hex[3:5], 16),
                int(color_hex[5:7], 16)
            ]
        
        # Filtrar detecciones
        filtered_detections = []
        
        for detection in detections_db:
            if detection['class'] == 'frame':  # Ignorar frames completos
                continue
                
            # Filtrar por color si se especifica
            if color_rgb is not None and 'dominant_color' in detection:
                # Calcular distancia euclidiana entre colores
                det_color = detection['dominant_color']
                color_distance = math.sqrt(
                    (color_rgb[0] - det_color[0])**2 +
                    (color_rgb[1] - det_color[1])**2 +
                    (color_rgb[2] - det_color[2])**2
                )
                # Descartar si el color es muy diferente (umbral 100)
                if color_distance > 100:
                    continue
            
            # Filtrar por fecha
            if start_date or end_date:
                if 'detection_time' not in detection:
                    continue
                
                detection_date = detection['detection_time'].split()[0]  # Obtener solo la fecha
                
                if start_date and detection_date < start_date:
                    continue
                if end_date and detection_date > end_date:
                    continue
            
            # Filtrar por rango de tiempo (hora del día)
            if time_range and 'detection_time' in detection:
                try:
                    start_time, end_time = time_range.split('-')
                    detection_time = detection['detection_time'].split()[1][:5]  # HH:MM
                    
                    if not (start_time <= detection_time <= end_time):
                        continue
                except:
                    pass  # Ignorar errores en formato de tiempo
            
            # Filtrar por tamaño del objeto
            if size_min and 'object_size' in detection:
                if float(detection['object_size']) < float(size_min):
                    continue
            
            if size_max and 'object_size' in detection:
                if float(detection['object_size']) > float(size_max):
                    continue
            
            # Si pasa todos los filtros, añadir a resultados
            filtered_detections.append(detection)
        
        return jsonify({
            'success': True,
            'count': len(filtered_detections),
            'detections': filtered_detections
        })
        
    except Exception as e:
        log_error("Error in forensic_search", e)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


# Función para procesar el video en un hilo separado
def process_video(video_path, model_path, threshold, show_tracks):
  global processing_status, detections_db, trajectory_data
  
  try:
      processing_status['is_processing'] = True
      processing_status['status'] = 'loading_model'
      processing_status['error'] = None
      processing_status['progress'] = 0
      processing_status['detections'] = []
      detections_db = []
      trajectory_data = {}
      
      logger.info(f"Starting video processing in background thread")
      logger.info(f"Loading model: {model_path}")
      
      try:
          model = YOLO(model_path)
          logger.info("Model loaded successfully")
      except Exception as e:
          log_error(f"Error loading model: {model_path}", e)
          processing_status['status'] = 'error'
          processing_status['error'] = f"Error loading model: {str(e)}"
          processing_status['is_processing'] = False
          return
      
      # Configurar el tracker
      tracker = "bytetrack.yaml" if show_tracks else None
      
      # Generar un ID único para esta detección
      detection_id = str(uuid.uuid4())
      output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{detection_id}.mp4")
      
      # Procesar el video
      logger.info(f"Processing video: {video_path}")
      processing_status['status'] = 'processing_video'
      
      # Abrir el video para obtener información
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
          log_error(f"Could not open video file: {video_path}")
          processing_status['status'] = 'error'
          processing_status['error'] = "Could not open video file"
          processing_status['is_processing'] = False
          return
      
      # Obtener propiedades del video
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = cap.get(cv2.CAP_PROP_FPS)
      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      
      processing_status['total_frames'] = total_frames
      
      # Configurar el escritor de video
      fourcc = cv2.VideoWriter_fourcc(*'avc1') 
      out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
      
      # Procesar cada frame
      frame_count = 0
      
      try:
          # Ejecutar el tracker en el video
          results = model.track(
              source=video_path,
              conf=threshold,  # Usar el valor directamente
              iou=0.5,
              show=False,
              stream=True,
              tracker=tracker,
              device='cpu'  # Forzar CPU para evitar problemas con CUDA
          )
          
          for result in results:
              try:
                  # Actualizar progreso
                  frame_count += 1
                  processing_status['current_frame'] = frame_count
                  processing_status['progress'] = int((frame_count / total_frames) * 100)
                  
                  # Obtener el frame original
                  orig_img = result.orig_img
                  
                  # Obtener las detecciones
                  boxes = result.boxes
                  
                  # Crear una copia del frame para anotaciones
                  annotated_frame = orig_img.copy()
                  
                  # Dibujar las detecciones
                  for box in boxes:
                      # Obtener coordenadas
                      x1, y1, x2, y2 = map(int, box.xyxy[0])
                      
                      # Calcular centro del objeto
                      x_center = (x1 + x2) / 2
                      y_center = (y1 + y2) / 2
                      
                      # Obtener clase y confianza
                      cls = int(box.cls[0])
                      conf = float(box.conf[0])
                      class_name = model.names[cls]
                      
                      # Obtener ID de tracking si está disponible
                      track_id = None
                      if hasattr(box, 'id') and box.id is not None:
                          track_id = int(box.id[0])
                      
                      # Dibujar rectángulo
                      color = (0, 255, 0)  # Verde por defecto
                      cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                      
                      # Dibujar etiqueta
                      label = f"{class_name} {conf:.2f}"
                      if track_id is not None:
                          label += f" ID:{track_id}"
                      
                      # Fondo para el texto
                      text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                      cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
                      
                      # Texto
                      cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                      
                      # Guardar datos de trayectoria si hay ID de seguimiento
                      if track_id is not None:
                          # Guardar la trayectoria antes de procesar la imagen
                          # para asegurarnos de que tenemos los datos incluso si falla el guardado de la imagen
                          save_trajectory_data(
                              track_id=track_id,
                              frame_number=frame_count,
                              x=x_center,
                              y=y_center,
                              class_name=class_name,
                              confidence=conf,
                              image_path="",  # Se actualizará después si se guarda la imagen
                              frame_width=width,
                              frame_height=height
                          )
                      
                      # Guardar la detección en la base de datos
                      if frame_count % 10 == 0:  # Guardar cada 10 frames para no sobrecargar
                          # Recortar la detección
                          detection_crop = orig_img[y1:y2, x1:x2]
                          if detection_crop.size > 0:
                              try:
                                  # Convertir de BGR a RGB
                                  rgb_crop = cv2.cvtColor(detection_crop, cv2.COLOR_BGR2RGB)
                                  
                                  # Guardar la imagen en la carpeta de detecciones
                                  detection_id = str(uuid.uuid4())
                                  detection_path = os.path.join(app.config['DETECTION_FOLDER'], f"{detection_id}.jpg")
                                  
                                  # Guardar con PIL
                                  pil_img = Image.fromarray(rgb_crop)
                                  pil_img.save(detection_path, quality=95)
                                  logger.info(f"Saved detection to {detection_path}")
                                  
                                  # Copiar a la carpeta estática para acceso web
                                  static_path = copy_to_static(detection_path)
                                  
                                  # Registrar la detección
                                  detection_data = {
                                      'id': detection_id,
                                      'image_base64': image_to_base64(detection_path),
                                      'x_center': x_center,  # Añadir coordenadas del centro
                                      'y_center': y_center,
                                      'filename': f"{detection_id}.jpg",
                                    # Añadir datos para el módulo forense
                                      'dominant_color': extract_dominant_color(detection_path),
                                      'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                      'object_size': (x2-x1) * (y2-y1),
                                      'class': class_name,
                                      'confidence': conf,
                                      'track_id': track_id,
                                      'frame': frame_count,
                                      'timestamp': datetime.now().isoformat(),
                                      'image_path': static_path or f"/detections/{detection_id}.jpg",
                                      'image_base64': image_to_base64(detection_path),
                                      'x_center': x_center,  # Añadir coordenadas del centro
                                      'y_center': y_center,
                                      'filename': f"{detection_id}.jpg"
                                  }
                                  detections_db.append(detection_data)
                                  processing_status['detections'].append(detection_data)
                                  
                                  # Actualizar la ruta de la imagen en los datos de trayectoria
                                  if track_id is not None:
                                      # Actualizar el último elemento añadido con la ruta de la imagen
                                      trajectory_data[track_id]['filenames'][-1] = static_path or f"/detections/{detection_id}.jpg"
                              except Exception as e:
                                  log_error(f"Failed to save detection image", e)
                  
                  # Guardar frame para la galería
                  if frame_count % 30 == 0:
                      try:
                          # Convertir de BGR a RGB
                          rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                          
                          # Guardar el frame
                          frame_id = str(uuid.uuid4())
                          frame_path = os.path.join(app.config['DETECTION_FOLDER'], f"{frame_id}.jpg")
                          
                          # Guardar con PIL
                          pil_img = Image.fromarray(rgb_frame)
                          pil_img.save(frame_path, quality=95)
                          logger.info(f"Saved gallery frame to {frame_path}")
                          
                          # Copiar a la carpeta estática para acceso web
                          static_path = copy_to_static(frame_path)
                          
                          # Registrar el frame
                          frame_data = {
                              'id': frame_id,
                              'class': 'frame',
                              'confidence': 1.0,
                              'track_id': None,
                              'frame': frame_count,
                              'timestamp': datetime.now().isoformat(),
                              'image_path': static_path or f"/detections/{frame_id}.jpg",
                              'image_base64': image_to_base64(frame_path),
                              'filename': f"{frame_id}.jpg"
                          }
                          detections_db.append(frame_data)
                          processing_status['detections'].append(frame_data)
                      except Exception as e:
                          log_error(f"Error saving gallery frame", e)
                  
                  # Escribir el frame anotado al video de salida
                  try:
                      out.write(annotated_frame)
                  except Exception as e:
                      log_error(f"Error writing frame to output video", e)
                  
                  # Mostrar progreso cada 100 frames
                  if frame_count % 100 == 0:
                      logger.info(f"Processed {frame_count}/{total_frames} frames")
              except Exception as frame_error:
                  log_error(f"Error processing frame {frame_count}", frame_error)
                  continue
          
      except Exception as e:
          log_error("Error during video processing", e)
          # Liberar recursos antes de devolver el error
          try:
              cap.release()
              out.release()
          except:
              pass
          processing_status['status'] = 'error'
          processing_status['error'] = f"Error during video processing: {str(e)}"
          processing_status['is_processing'] = False
          return
      
      # Liberar recursos
      try:
          cap.release()
          out.release()
      except Exception as e:
          log_error("Error releasing resources", e)
      
      logger.info(f"Video processing completed. Output saved to {output_path}")
      
      # Renombrar el video para facilitar su acceso
      simple_video_name = copy_and_rename_output_video(output_path)
      if simple_video_name:
          logger.info(f"Video renombrado a: {simple_video_name}")
          processing_status['simple_video_name'] = simple_video_name
      
      # Verificar que el archivo de video existe y tiene un tamaño adecuado
      if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
          logger.info(f"Video file created successfully: {output_path} ({os.path.getsize(output_path)} bytes)")
      else:
          logger.error(f"Video file was not created properly: {output_path}")
          # Intentar copiar el video original como fallback
          try:
              fallback_output = os.path.join(app.config['OUTPUT_FOLDER'], f"fallback_{detection_id}.mp4")
              shutil.copy2(video_path, fallback_output)
              logger.info(f"Copied original video as fallback: {fallback_output}")
              output_path = fallback_output
          except Exception as e:
              log_error(f"Failed to create fallback video", e)

      # Copiar el video de salida a la carpeta estática
      static_video_path = copy_to_static(output_path, 'videos')
      if not static_video_path:
          logger.error(f"Failed to copy video to static folder: {output_path}")
          # Intentar copiar directamente
          try:
              filename = f"{str(uuid.uuid4())}{os.path.splitext(output_path)[1]}"
              static_path = os.path.join(app.config['STATIC_FOLDER'], 'videos', filename)
              shutil.copy2(output_path, static_path)
              static_video_path = f"/static/videos/{filename}"
              logger.info(f"Direct copy to static folder successful: {static_path}")
          except Exception as e:
              log_error(f"Direct copy to static folder failed", e)
              static_video_path = f"/outputs/{os.path.basename(output_path)}"

      # Actualizar estado
      processing_status['status'] = 'completed'
      processing_status['is_processing'] = False

      # Guardar el video con un nombre predecible
      renamed_path, renamed_filename = save_processed_video(output_path, app.config['OUTPUT_FOLDER'])
      logger.info(f"Video guardado con nombre predecible: {renamed_path}, {renamed_filename}")

      # Asegurarse de que el archivo existe
      if os.path.exists(renamed_path):
          file_size = os.path.getsize(renamed_path)
          logger.info(f"Video file exists: {renamed_path}, size: {file_size} bytes")
      else:
          logger.error(f"Renamed video file does not exist: {renamed_path}")

      # Actualizar las rutas en el estado
      static_video_path = f"/outputs/{renamed_filename}"
      processing_status['output_video'] = static_video_path
      processing_status['output_video_filename'] = renamed_filename

      # Registrar información adicional para depuración
      processing_status['debug_info'] = {
          'original_path': output_path,
          'renamed_path': renamed_path,
          'static_path': static_video_path,
          'exists': os.path.exists(renamed_path),
          'size': os.path.getsize(renamed_path) if os.path.exists(renamed_path) else 0
      }

      # Guardar datos de trayectoria en un archivo JSON
      try:
          trajectory_file = os.path.join(app.config['TRAJECTORY_FOLDER'], f"trajectory_{detection_id}.json")
          with open(trajectory_file, 'w') as f:
              json.dump(trajectory_data, f)
          logger.info(f"Trajectory data saved to {trajectory_file}")
      except Exception as e:
          log_error(f"Error saving trajectory data", e)

      logger.info(f"Processing completed. Video available at: {static_video_path}")
      
  except Exception as e:
      log_error("Unexpected error in process_video", e)
      processing_status['status'] = 'error'
      processing_status['error'] = f"Unexpected error: {str(e)}"
      processing_status['is_processing'] = False

@app.route('/detect', methods=['POST'])
def detect():
  try:
      global processing_status
      
      # Verificar si ya hay un procesamiento en curso
      if processing_status['is_processing']:
          return jsonify({
              'success': False,
              'error': 'A video is already being processed. Please wait for it to complete.'
          })
      
      data = request.json
      video_path = data.get('video_path')
      model_path = data.get('model_path')
      threshold = float(data.get('threshold', 0.5))
      show_tracks = data.get('show_tracks', True)
      
      if not video_path or not os.path.exists(video_path):
          return jsonify({'success': False, 'error': 'Video file not found'})
      
      if not model_path or not os.path.exists(model_path):
          return jsonify({'success': False, 'error': 'Model file not found'})
      
      # Iniciar procesamiento en un hilo separado
      processing_thread = threading.Thread(
          target=process_video,
          args=(video_path, model_path, threshold, show_tracks)
      )
      processing_thread.daemon = True
      processing_thread.start()
      
      return jsonify({
          'success': True,
          'message': 'Video processing started in background',
          'status_url': '/processing_status'
      })
      
  except Exception as e:
      log_error("Error in detect", e)
      return jsonify({
          'success': False,
          'error': str(e),
          'traceback': traceback.format_exc()
      })

@app.route('/processing_status')
def get_processing_status():
  global processing_status
  return jsonify(processing_status)

@app.route('/detections/<filename>')
def detection_file(filename):
  logger.info(f"Serving detection file: {filename}")
  try:
      # Asegurarse de que el navegador no almacene en caché las imágenes
      response = send_from_directory(app.config['DETECTION_FOLDER'], filename)
      response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
      response.headers['Pragma'] = 'no-cache'
      response.headers['Expires'] = '0'
      return response
  except Exception as e:
      log_error(f"Error serving detection file: {filename}", e)
      # Redirigir a la imagen de marcador de posición
      return send_from_directory(app.config['STATIC_FOLDER'], 'placeholder.jpg')

@app.route('/outputs/<filename>')
def output_file(filename):
    logger.info(f"Serving output file: {filename}")
    try:
        # Obtener la ruta completa del archivo
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        # Registrar información sobre el archivo
        file_size = os.path.getsize(file_path)
        logger.info(f"File exists: {file_path}, size: {file_size} bytes")
        
        # Determinar el tipo MIME basado en la extensión del archivo
        mime_type = 'video/mp4'
        if filename.lower().endswith('.avi'):
            mime_type = 'video/x-msvideo'
        elif filename.lower().endswith('.mov'):
            mime_type = 'video/quicktime'

        # Enviar el archivo con MIME type correcto y headers actualizados
        response = send_from_directory(
            app.config['OUTPUT_FOLDER'], 
            filename,
            mimetype=mime_type,
            as_attachment=False,
            conditional=True
        )
        
        # Añadir headers para mejorar la compatibilidad de streaming
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'public, max-age=300'  # Caché por 5 minutos
        response.headers['Content-Disposition'] = f'inline; filename="{filename}"'
        
        return response
    except Exception as e:
        log_error(f"Error serving output file: {filename}", e)
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
  logger.info(f"Serving static file: {filename}")
  return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.route('/dashboard_stats')
def dashboard_stats():
  try:
      # Calcular estadísticas
      class_distribution = {}
      avg_confidence = {}
      time_series = {}
      track_distribution = {}
      
      for detection in detections_db:
          # Distribución de clases
          class_name = detection['class']
          if class_name not in class_distribution:
              class_distribution[class_name] = 0
          class_distribution[class_name] += 1
          
          # Confianza promedio por clase
          if class_name not in avg_confidence:
              avg_confidence[class_name] = []
          avg_confidence[class_name].append(detection['confidence'])
          
          # Serie temporal
          date = detection['timestamp'].split('T')[0]
          if date not in time_series:
              time_series[date] = 0
          time_series[date] += 1
          
          # Distribución de track IDs
          if detection['track_id'] is not None:
              track_id = f"ID {detection['track_id']}"
              if track_id not in track_distribution:
                  track_distribution[track_id] = 0
              track_distribution[track_id] += 1
      
      # Calcular promedios de confianza
      for cls in avg_confidence:
          if avg_confidence[cls]:  # Verificar que la lista no esté vacía
              avg_confidence[cls] = sum(avg_confidence[cls]) / len(avg_confidence[cls])
          else:
              avg_confidence[cls] = 0
      
      return jsonify({
          'class_distribution': class_distribution,
          'avg_confidence': avg_confidence,
          'time_series': time_series,
          'track_distribution': track_distribution
      })
      
  except Exception as e:
      log_error("Error in dashboard_stats", e)
      return jsonify({'error': str(e)})

@app.route('/search_detections')
def search_detections():
  try:
      class_filter = request.args.get('class', '').lower()
      track_id = request.args.get('track_id', '')
      
      filtered_detections = detections_db
      
      # Filtrar por clase
      if class_filter:
          filtered_detections = [d for d in filtered_detections if class_filter in d['class'].lower()]
      
      # Filtrar por track ID
      if track_id:
          try:
              track_id_int = int(track_id)
              filtered_detections = [d for d in filtered_detections if d['track_id'] == track_id_int]
          except ValueError:
              pass
      
      return jsonify({
          'detections': filtered_detections
      })
      
  except Exception as e:
      log_error("Error in search_detections", e)
      return jsonify({'error': str(e)})

# Nueva ruta para obtener datos de trayectorias
@app.route('/trajectory_data')
def get_trajectory_data():
    """
    Endpoint para obtener los datos de trayectoria de todos los objetos detectados.
    Devuelve un JSON con los datos de trayectoria organizados por ID de seguimiento.
    """
    global trajectory_data
    
    try:
        # Verificar si hay datos de trayectoria
        if not trajectory_data or len(trajectory_data) == 0:
            logger.warning("No hay datos de trayectoria disponibles")
            
            # Intentar cargar datos de trayectoria desde archivos guardados
            trajectory_files = [f for f in os.listdir(app.config['TRAJECTORY_FOLDER']) if f.endswith('.json')]
            
            if trajectory_files:
                # Ordenar por fecha de modificación (más reciente primero)
                trajectory_files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['TRAJECTORY_FOLDER'], x)), reverse=True)
                
                # Cargar el archivo más reciente
                latest_file = trajectory_files[0]
                file_path = os.path.join(app.config['TRAJECTORY_FOLDER'], latest_file)
                
                logger.info(f"Cargando datos de trayectoria desde archivo: {file_path}")
                
                try:
                    with open(file_path, 'r') as f:
                        trajectory_data = json.load(f)
                    
                    logger.info(f"Datos de trayectoria cargados correctamente: {len(trajectory_data)} objetos")
                except Exception as e:
                    logger.error(f"Error al cargar archivo de trayectoria: {e}")
                    return jsonify({
                        'success': False,
                        'error': f"Error al cargar archivo de trayectoria: {str(e)}",
                        'trajectory_count': 0,
                        'trajectories': {}
                    })
            else:
                logger.warning("No se encontraron archivos de trayectoria")
                return jsonify({
                    'success': False,
                    'error': "No hay datos de trayectoria disponibles",
                    'trajectory_count': 0,
                    'trajectories': {}
                })
        
        # Calcular métricas adicionales para cada trayectoria
        enriched_data = {}
        
        for track_id, data in trajectory_data.items():
            # Verificar que los datos tengan la estructura esperada
            if not isinstance(data, dict) or 'positions' not in data:
                logger.warning(f"Datos de trayectoria con formato incorrecto para track_id {track_id}")
                continue
                
            # Copiar datos básicos
            enriched_data[track_id] = {
                'class': data.get('class', 'unknown'),
                'frames': data.get('frames', []),
                'positions': data.get('normalized_positions', data.get('positions', [])),
                'filenames': data.get('filenames', []),
                'confidences': data.get('confidences', []),
                'timestamps': data.get('timestamps', []),
                'metrics': {}
            }
            
            # Calcular métricas si hay suficientes puntos
            positions = data.get('positions', [])
            if len(positions) >= 2:
                # Calcular distancia total recorrida
                total_distance = 0
                for i in range(1, len(positions)):
                    p1 = positions[i-1]
                    p2 = positions[i]
                    
                    # Verificar que las posiciones tengan las coordenadas esperadas
                    if not all(k in p1 for k in ['x', 'y']) or not all(k in p2 for k in ['x', 'y']):
                        continue
                        
                    distance = math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
                    total_distance += distance
                
                # Calcular velocidad promedio (píxeles/frame)
                avg_velocity = total_distance / (len(positions) - 1)
                
                # Calcular cambios de dirección
                direction_changes = 0
                prev_direction = None
                
                for i in range(1, len(positions)):
                    p1 = positions[i-1]
                    p2 = positions[i]
                    
                    # Verificar que las posiciones tengan las coordenadas esperadas
                    if not all(k in p1 for k in ['x', 'y']) or not all(k in p2 for k in ['x', 'y']):
                        continue
                    
                    # Calcular dirección actual (ángulo en radianes)
                    dx = p2['x'] - p1['x']
                    dy = p2['y'] - p1['y']
                    
                    if dx == 0 and dy == 0:
                        continue  # Evitar división por cero
                        
                    current_direction = math.atan2(dy, dx)
                    
                    # Detectar cambio de dirección
                    if prev_direction is not None:
                        angle_diff = abs(current_direction - prev_direction)
                        # Normalizar a [0, π]
                        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                        
                        # Considerar un cambio significativo si es mayor a 45 grados
                        if angle_diff > math.pi/4:
                            direction_changes += 1
                    
                    prev_direction = current_direction
                
                # Guardar métricas calculadas
                frames = data.get('frames', [])
                enriched_data[track_id]['metrics'] = {
                    'total_distance': total_distance,
                    'avg_velocity': avg_velocity,
                    'direction_changes': direction_changes,
                    'duration_frames': len(frames) if frames else len(positions),
                    'start_frame': frames[0] if frames else 0,
                    'end_frame': frames[-1] if frames else len(positions) - 1
                }
        
        logger.info(f"Enviando datos de trayectoria: {len(enriched_data)} objetos")
        
        return jsonify({
            'success': True,
            'trajectory_count': len(enriched_data),
            'trajectories': enriched_data
        })
        
    except Exception as e:
        log_error("Error in get_trajectory_data", e)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'trajectory_count': 0,
            'trajectories': {}
        })

@app.route('/trajectory_analysis/<int:track_id>')
def trajectory_analysis(track_id):
    """
    Endpoint para analizar una trayectoria específica.
    Devuelve análisis detallado de la trayectoria del objeto con el ID especificado.
    """
    global trajectory_data
    
    try:
        track_id_str = str(track_id)
        
        if track_id_str not in trajectory_data:
            return jsonify({
                'success': False,
                'error': f'No trajectory data found for track_id {track_id}'
            })
        
        data = trajectory_data[track_id_str]
        
        # Calcular velocidad instantánea
        velocities = []
        for i in range(1, len(data['positions'])):
            p1 = data['positions'][i-1]
            p2 = data['positions'][i]
            t1 = data['timestamps'][i-1]
            t2 = data['timestamps'][i]
            
            distance = math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
            time_diff = t2 - t1
            
            if time_diff > 0:
                velocity = distance / time_diff
                velocities.append({
                    'frame': data['frames'][i],
                    'velocity': velocity
                })
        
        # Calcular aceleración
        accelerations = []
        for i in range(1, len(velocities)):
            v1 = velocities[i-1]['velocity']
            v2 = velocities[i]['velocity']
            t1 = data['timestamps'][i]
            t2 = data['timestamps'][i+1]
            
            time_diff = t2 - t1
            
            if time_diff > 0:
                acceleration = (v2 - v1) / time_diff
                accelerations.append({
                    'frame': data['frames'][i+1],
                    'acceleration': acceleration
                })
        
        # Detectar comportamientos anómalos
        anomalies = []
        
        # 1. Detección de paradas repentinas
        for i in range(1, len(velocities)):
            if velocities[i-1]['velocity'] > 0.1 and velocities[i]['velocity'] < 0.01:
                anomalies.append({
                    'type': 'sudden_stop',
                    'frame': velocities[i]['frame'],
                    'description': 'Parada repentina detectada'
                })
        
        # 2. Detección de cambios bruscos de dirección
        for i in range(2, len(data['positions'])):
            p1 = data['positions'][i-2]
            p2 = data['positions'][i-1]
            p3 = data['positions'][i]
            
            # Vectores de dirección
            v1 = (p2['x'] - p1['x'], p2['y'] - p1['y'])
            v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'])
            
            # Normalizar vectores
            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len_v1 > 0 and len_v2 > 0:
                v1_norm = (v1[0]/len_v1, v1[1]/len_v1)
                v2_norm = (v2[0]/len_v2, v2[1]/len_v2)
                
                # Producto escalar (coseno del ángulo)
                dot_product = v1_norm[0]*v2_norm[0] + v1_norm[1]*v2_norm[1]
                
                # Ángulo en radianes
                angle = math.acos(max(-1, min(1, dot_product)))
                
                # Convertir a grados
                angle_deg = angle * 180 / math.pi
                
                # Considerar cambio brusco si es mayor a 90 grados
                if angle_deg > 90:
                    anomalies.append({
                        'type': 'sharp_turn',
                        'frame': data['frames'][i],
                        'angle': angle_deg,
                        'description': f'Giro brusco de {angle_deg:.1f} grados'
                    })
        
        return jsonify({
            'success': True,
            'track_id': track_id,
            'class': data['class'],
            'frame_count': len(data['frames']),
            'positions': data['normalized_positions'],
            'velocities': velocities,
            'accelerations': accelerations,
            'anomalies': anomalies,
            'frames': data['frames'],
            'filenames': data['filenames']
        })
        
    except Exception as e:
        log_error(f"Error in trajectory_analysis for track_id {track_id}", e)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/heatmap_data')
def heatmap_data():
    """
    Endpoint para generar datos de mapa de calor basado en las trayectorias.
    Devuelve una matriz de intensidad para visualizar áreas de mayor actividad.
    """
    global trajectory_data
    
    try:
        # Parámetros de configuración
        width = int(request.args.get('width', 800))
        height = int(request.args.get('height', 600))
        class_filter = request.args.get('class', 'all')
        
        # Inicializar matriz de intensidad
        intensity_matrix = [[0 for _ in range(width)] for _ in range(height)]
        
        # Acumular puntos de todas las trayectorias
        for track_id, data in trajectory_data.items():
            # Aplicar filtro de clase si es necesario
            if class_filter != 'all' and data['class'] != class_filter:
                continue
                
            for pos in data['normalized_positions']:
                # Convertir coordenadas normalizadas a píxeles
                x = int(pos['x'] * width)
                y = int(pos['y'] * height)
                
                # Asegurarse de que las coordenadas están dentro de los límites
                if 0 <= x < width and 0 <= y < height:
                    # Aplicar kernel gaussiano para suavizar
                    kernel_size = 5
                    for i in range(-kernel_size, kernel_size + 1):
                        for j in range(-kernel_size, kernel_size + 1):
                            nx, ny = x + i, y + j
                            if 0 <= nx < width and 0 <= ny < height:
                                # Calcular intensidad basada en la distancia al centro
                                distance = math.sqrt(i*i + j*j)
                                intensity = math.exp(-(distance*distance) / (2 * (kernel_size/2)**2))
                                intensity_matrix[ny][nx] += intensity
        
        # Normalizar la matriz de intensidad
        max_intensity = max(max(row) for row in intensity_matrix)
        if max_intensity > 0:
            normalized_matrix = [[cell / max_intensity for cell in row] for row in intensity_matrix]
        else:
            normalized_matrix = intensity_matrix
        
        return jsonify({
            'success': True,
            'width': width,
            'height': height,
            'intensity_matrix': normalized_matrix
        })
        
    except Exception as e:
        log_error("Error in heatmap_data", e)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
@app.route('/upload_form')
def upload_form():
    return render_template('upload_form.html')

@app.route('/upload_simple', methods=['POST'])
def upload_simple():
    try:
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file:
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return f'File uploaded successfully to {filepath}'
    except Exception as e:
        log_error("Error in upload_simple", e)
        return f'Error: {str(e)}'

@app.route('/upload_model_simple', methods=['POST'])
def upload_model_simple():
    try:
        if 'model' not in request.files:
            return 'No model file part'
        
        file = request.files['model']
        if file.filename == '':
            return 'No selected file'
        
        if not file.filename.lower().endswith('.pt'):
            return 'File must be a YOLOv8 model (.pt)'
        
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
        file.save(filepath)
        return f'Model uploaded successfully to {filepath}'
    except Exception as e:
        log_error("Error in upload_model_simple", e)
        return f'Error: {str(e)}'
@app.route('/search_by_color')
def search_by_color():
    try:
        # Obtener parámetros de la consulta
        color_hex = request.args.get('color', '')
        tolerance = int(request.args.get('tolerance', 30))
        date_from = request.args.get('date_from', '')
        date_to = request.args.get('date_to', '')
        time_from = request.args.get('time_from', '')
        time_to = request.args.get('time_to', '')
        
        logger.info(f"Búsqueda por color: {color_hex}, tolerancia: {tolerance}%, fecha desde: {date_from}, fecha hasta: {date_to}, hora desde: {time_from}, hora hasta: {time_to}")
        
        if not color_hex:
            return jsonify({
                'success': False,
                'error': 'No se proporcionó un color para la búsqueda',
                'detections': []
            })
        
        # Convertir color hexadecimal a RGB
        try:
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            target_color = (r, g, b)
            
            # Convertir a HSV para mejor comparación de colores
            target_hsv = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            logger.info(f"Color objetivo RGB: {target_color}, HSV: {target_hsv}")
        except Exception as e:
            log_error(f"Error al convertir color hexadecimal: {color_hex}", e)
            return jsonify({
                'success': False,
                'error': f'Color inválido: {color_hex}',
                'detections': []
            })
        
        # Filtrar por fecha y hora si se proporcionaron
        filtered_detections = detections_db.copy()
        
        if date_from:
            filtered_detections = [d for d in filtered_detections if d['timestamp'].split('T')[0] >= date_from]
        
        if date_to:
            filtered_detections = [d for d in filtered_detections if d['timestamp'].split('T')[0] <= date_to]
        
        if time_from or time_to:
            # Extraer la parte de la hora de la marca de tiempo
            pattern = re.compile(r'T(\d{2}:\d{2}:\d{2})')
            
            if time_from:
                filtered_detections = [d for d in filtered_detections if 
                                      pattern.search(d['timestamp']) and 
                                      pattern.search(d['timestamp']).group(1) >= time_from]
            
            if time_to:
                filtered_detections = [d for d in filtered_detections if 
                                      pattern.search(d['timestamp']) and 
                                      pattern.search(d['timestamp']).group(1) <= time_to]
        
        # Excluir frames de galería
        filtered_detections = [d for d in filtered_detections if d['class'] != 'frame']
        
        # Analizar cada detección para encontrar coincidencias de color
        color_matches = []
        
        for detection in filtered_detections:
            try:
                # Obtener la ruta de la imagen
                image_path = detection['image_path']
                if image_path.startswith('/'):
                    # Convertir ruta relativa a absoluta
                    if image_path.startswith('/static/'):
                        image_path = os.path.join(app.config['STATIC_FOLDER'], image_path[8:])
                    elif image_path.startswith('/detections/'):
                        image_path = os.path.join(app.config['DETECTION_FOLDER'], image_path[12:])
                    else:
                        # Intentar buscar en la carpeta de detecciones
                        possible_path = os.path.join(app.config['DETECTION_FOLDER'], os.path.basename(image_path))
                        if os.path.exists(possible_path):
                            image_path = possible_path
                
                # Verificar si la imagen existe
                if not os.path.exists(image_path):
                    logger.warning(f"Imagen no encontrada: {image_path}")
                    continue
                
                # Abrir la imagen y calcular el color dominante
                img = Image.open(image_path)
                img = img.convert('RGB')
                
                # Calcular estadísticas de color
                stat = ImageStat.Stat(img)
                avg_color_rgb = stat.mean
                
                # Convertir a HSV para mejor comparación
                avg_color_hsv = colorsys.rgb_to_hsv(
                    avg_color_rgb[0]/255, 
                    avg_color_rgb[1]/255, 
                    avg_color_rgb[2]/255
                )
                
                # Calcular diferencia de color (distancia en espacio HSV)
                # Dar más peso al tono (H) y la saturación (S) que al valor (V)
                h_diff = min(abs(target_hsv[0] - avg_color_hsv[0]), 1 - abs(target_hsv[0] - avg_color_hsv[0])) * 2
                s_diff = abs(target_hsv[1] - avg_color_hsv[1])
                v_diff = abs(target_hsv[2] - avg_color_hsv[2]) * 0.5
                
                # Calcular coincidencia como porcentaje (100% - diferencia)
                color_diff = (h_diff + s_diff + v_diff) / 3.5 * 100
                color_match = max(0, 100 - color_diff)
                
                # Filtrar por tolerancia
                if color_match >= (100 - tolerance):
                    # Añadir información de coincidencia de color
                    detection_copy = detection.copy()
                    detection_copy['color_match'] = round(color_match, 1)
                    detection_copy['avg_color_rgb'] = [round(c) for c in avg_color_rgb]
                    detection_copy['avg_color_hex'] = '#{:02x}{:02x}{:02x}'.format(
                        round(avg_color_rgb[0]), 
                        round(avg_color_rgb[1]), 
                        round(avg_color_rgb[2])
                    )
                    
                    color_matches.append(detection_copy)
                    
            except Exception as e:
                log_error(f"Error al analizar color de la imagen: {image_path}", e)
                continue
        
        # Ordenar por coincidencia de color (mayor primero)
        color_matches.sort(key=lambda x: x['color_match'], reverse=True)
        
        logger.info(f"Búsqueda por color completada. Encontradas {len(color_matches)} coincidencias.")
        
        return jsonify({
            'success': True,
            'detections': color_matches,
            'target_color': {
                'rgb': target_color,
                'hex': f'#{color_hex}',
                'hsv': [round(target_hsv[0] * 360), round(target_hsv[1] * 100), round(target_hsv[2] * 100)]
            },
            'total_matches': len(color_matches)
        })
        
    except Exception as e:
        log_error("Error en search_by_color", e)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'detections': []
        })
if __name__ == '__main__':
  # Crear imagen de marcador de posición al inicio
  placeholder_path = os.path.join(app.config['STATIC_FOLDER'], 'placeholder.jpg')
  if not os.path.exists(placeholder_path):
      try:
          img = Image.new('RGB', (400, 300), color=(240, 240, 240))
          d = ImageDraw.Draw(img)
          d.rectangle([0, 0, 400, 300], outline=(200, 200, 200), width=2)
          d.text((150, 140), "Imagen no disponible", fill=(100, 100, 100))
          img.save(placeholder_path)
          logger.info(f"Created placeholder image at {placeholder_path}")
      except Exception as e:
          log_error(f"Failed to create placeholder image", e)
          # Crear una imagen en blanco como fallback
          try:
              blank_img = Image.new('RGB', (400, 300), color=(240, 240, 240))
              blank_img.save(placeholder_path)
              logger.info(f"Created blank placeholder image at {placeholder_path}")
          except Exception as e2:
              log_error(f"Failed to create blank placeholder image", e2)
  
  logger.info("Starting Object Detection and Tracking Server")
  app.run(debug=True, host='0.0.0.0', port=5000)
