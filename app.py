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
from PIL import Image, ImageDraw
import base64
import io
import threading
import time
import sys

# Configurar logging
logging.basicConfig(
  level=logging.DEBUG,  # Cambiar a DEBUG para más información
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  handlers=[
      logging.FileHandler("app.log"),
      logging.StreamHandler()
  ]
)
logger = logging.getLogger(__name__)

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

# Crear directorios necesarios con rutas absolutas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'videos'), exist_ok=True)

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

# Almacenamiento en memoria para detecciones
detections_db = []

# Estado de procesamiento
processing_status = {
  'is_processing': False,
  'progress': 0,
  'total_frames': 0,
  'current_frame': 0,
  'status': 'idle',
  'error': None,
  'output_video': None,
  'detections': []
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

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
  try:
      if 'video' not in request.files:
          return jsonify({'success': False, 'error': 'No video file provided'})
      
      file = request.files['video']
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
      
      return jsonify({
          'success': True,
          'video_path': filepath,
          'message': 'Video uploaded successfully'
      })
      
  except Exception as e:
      log_error("Error in upload_file", e)
      return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_model', methods=['POST'])
def upload_model():
  try:
      if 'model' not in request.files:
          return jsonify({'success': False, 'error': 'No model file provided'})
      
      file = request.files['model']
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
      
      return jsonify({
          'success': True,
          'model_path': filepath,
          'message': 'Model uploaded successfully'
      })
      
  except Exception as e:
      log_error("Error in upload_model", e)
      return jsonify({'success': False, 'error': str(e)})

# Función para procesar el video en un hilo separado
def process_video(video_path, model_path, threshold, show_tracks):
  global processing_status, detections_db
  
  try:
      processing_status['is_processing'] = True
      processing_status['status'] = 'loading_model'
      processing_status['error'] = None
      processing_status['progress'] = 0
      processing_status['detections'] = []
      detections_db = []
      
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
                                      'class': class_name,
                                      'confidence': conf,
                                      'track_id': track_id,
                                      'frame': frame_count,
                                      'timestamp': datetime.now().isoformat(),
                                      'image_path': static_path or f"/detections/{detection_id}.jpg",
                                      'image_base64': image_to_base64(detection_path)
                                  }
                                  detections_db.append(detection_data)
                                  processing_status['detections'].append(detection_data)
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
                              'image_base64': image_to_base64(frame_path)
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
  logger.info("Starting Object Detection and Tracking Server")
