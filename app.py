"""
Whisper Transcription Web Application
Веб-додаток для транскрипції аудіо/відео файлів з точними таймкодами
"""

import os
import sys
import whisper
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import time
from pathlib import Path
import subprocess
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mkv', 'flac', 'm4a', 'ogg', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

progress_data = {}
models_cache = {}

def allowed_file(filename):
    """Перевіряє чи дозволений формат файлу"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path, audio_path):
    """Витягує аудіо з відео файлу"""
    try:
        logger.info(f"Конвертація відео: {video_path} -> {audio_path}")
        result = subprocess.run([
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', 
            audio_path, '-y'
        ], check=True, capture_output=True, text=True)
        logger.info("Конвертація успішна")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Помилка конвертації: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg не знайдено. Встановіть FFmpeg: sudo apt-get install ffmpeg")
        return False

def load_whisper_model(model_size="base"):
    """Завантажує модель Whisper (з кешуванням)"""
    try:
        if model_size not in models_cache:
            logger.info(f"Завантаження моделі Whisper: {model_size}")
            models_cache[model_size] = whisper.load_model(model_size)
            logger.info(f"Модель {model_size} завантажена успішно")
        return models_cache[model_size]
    except Exception as e:
        logger.error(f"Помилка завантаження моделі: {e}")
        raise

def detect_speech_boundaries(audio_path, segments):
    """
    Визначає точні межі мовлення за допомогою аналізу енергії аудіо
    """
    try:
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        frame_length = 2048
        hop_length = 512
        
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        times = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length)
        
        threshold = np.percentile(energy, 20)
        
        refined_segments = []
        
        for segment in segments:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            if not text:
                continue
            
            search_window_start = max(0, start - 0.5)
            search_window_end = min(len(audio) / sr, end + 0.5)
            
            start_idx = np.searchsorted(times, search_window_start)
            end_idx = np.searchsorted(times, search_window_end)
            
            segment_energy = energy[start_idx:end_idx]
            segment_times = times[start_idx:end_idx]
            
            if len(segment_energy) == 0:
                refined_segments.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
                continue
            
            speech_mask = segment_energy > threshold
            speech_indices = np.where(speech_mask)[0]
            
            if len(speech_indices) > 0:
                actual_start = segment_times[speech_indices[0]]
                actual_end = segment_times[speech_indices[-1]]
                
                actual_start = max(start - 0.3, actual_start)
                actual_end = min(end + 0.2, actual_end + 0.3)
            else:
                actual_start = start
                actual_end = end
            
            if actual_end - actual_start < 0.2:
                actual_end = actual_start + 0.2
            
            refined_segments.append({
                'start': actual_start,
                'end': actual_end,
                'text': text
            })
        
        for i in range(len(refined_segments) - 1):
            if refined_segments[i]['end'] > refined_segments[i + 1]['start']:
                gap = (refined_segments[i + 1]['start'] + refined_segments[i]['end']) / 2
                refined_segments[i]['end'] = gap - 0.05
                refined_segments[i + 1]['start'] = gap + 0.05
        
        logger.info(f"Визначено точні межі мовлення для {len(refined_segments)} сегментів")
        return refined_segments
        
    except ImportError:
        logger.warning("librosa не встановлено. Використовується базовий алгоритм")
        return basic_speech_detection(segments)
    except Exception as e:
        logger.error(f"Помилка визначення меж мовлення: {e}")
        return basic_speech_detection(segments)

def basic_speech_detection(segments):
    """
    Базовий алгоритм коригування таймкодів без librosa
    """
    refined_segments = []
    
    for segment in segments:
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '').strip()
        
        if not text:
            continue
        
        duration = end - start
        words_count = len(text.split())
        
        if duration < 0.5:
            adjusted_start = start + 0.05
            adjusted_end = end
        elif words_count <= 2 and duration > 2:
            adjusted_start = start + 0.15
            adjusted_end = start + min(duration * 0.6, 1.5)
        else:
            adjusted_start = start + 0.1
            adjusted_end = end - 0.05
        
        adjusted_end = max(adjusted_start + 0.2, adjusted_end)
        
        refined_segments.append({
            'start': adjusted_start,
            'end': adjusted_end,
            'text': text
        })
    
    return refined_segments

def generate_srt(segments):
    """Генерує SRT формат субтитрів"""
    srt_lines = []
    
    for i, segment in enumerate(segments, 1):
        start = format_timestamp_srt(segment['start'])
        end = format_timestamp_srt(segment['end'])
        text = segment['text']
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")
    
    return '\n'.join(srt_lines)

def format_timestamp_srt(seconds):
    """Форматує секунди в SRT формат (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_audio(file_path, language, model_size, task_id):
    """Функція транскрипції з прогресом"""
    try:
        logger.info(f"Початок транскрипції: {file_path}, мова: {language}, модель: {model_size}")
        
        progress_data[task_id] = {
            'status': 'loading_model',
            'progress': 10,
            'message': 'Завантаження моделі...'
        }
        
        model = load_whisper_model(model_size)
        
        progress_data[task_id] = {
            'status': 'processing',
            'progress': 30,
            'message': 'Обробка аудіо...'
        }
        
        transcribe_options = {
            'verbose': False,
            'task': 'transcribe',
            'word_timestamps': False
        }
        
        if language and language != 'auto':
            transcribe_options['language'] = language
        
        result = model.transcribe(file_path, **transcribe_options)
        
        logger.info(f"Транскрипція завершена. Знайдено {len(result.get('segments', []))} сегментів")
        
        progress_data[task_id] = {
            'status': 'refining',
            'progress': 70,
            'message': 'Визначення меж мовлення...'
        }
        
        raw_segments = result.get('segments', [])
        refined_segments = detect_speech_boundaries(file_path, raw_segments)
        
        progress_data[task_id] = {
            'status': 'formatting',
            'progress': 90,
            'message': 'Форматування результату...'
        }
        
        full_text = result['text'].strip()
        
        timestamped_text = []
        for segment in refined_segments:
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            text = segment['text']
            
            timestamped_text.append(f"[{start_time} → {end_time}] {text}")
        
        timestamped_output = '\n'.join(timestamped_text) if timestamped_text else full_text
        
        srt_content = generate_srt(refined_segments)
        
        detected_language = result.get('language', 'невідома')
        
        progress_data[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Готово!',
            'result': {
                'full_text': full_text,
                'timestamped_text': timestamped_output,
                'srt_subtitles': srt_content,
                'detected_language': detected_language,
                'segments_count': len(refined_segments)
            }
        }
        
        logger.info(f"Транскрипція успішно завершена для задачі {task_id}")
        
    except Exception as e:
        logger.error(f"Помилка транскрипції: {e}", exc_info=True)
        progress_data[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Помилка: {str(e)}'
        }
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Файл видалено: {file_path}")
        except Exception as e:
            logger.error(f"Помилка видалення файлу: {e}")

def format_timestamp(seconds):
    """Форматує секунди в HH:MM:SS формат"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не знайдено'}), 400
        
        file = request.files['file']
        language = request.form.get('language', 'uk')
        model_size = request.form.get('model_size', 'base')
        
        if file.filename == '':
            return jsonify({'error': 'Файл не обрано'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Непідтримуваний формат файлу'}), 400
        
        task_id = str(int(time.time() * 1000))
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)
        
        logger.info(f"Файл збережено: {file_path}")
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext in {'mp4', 'avi', 'mkv', 'webm'}:
            audio_path = file_path.rsplit('.', 1)[0] + '.wav'
            progress_data[task_id] = {
                'status': 'converting',
                'progress': 5,
                'message': 'Конвертація відео в аудіо...'
            }
            if extract_audio(file_path, audio_path):
                os.remove(file_path)
                file_path = audio_path
            else:
                return jsonify({'error': 'Помилка конвертації відео. Перевірте чи встановлений FFmpeg.'}), 500
        
        thread = threading.Thread(
            target=transcribe_audio,
            args=(file_path, language, model_size, task_id),
            daemon=True
        )
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    except Exception as e:
        logger.error(f"Помилка при завантаженні: {e}", exc_info=True)
        return jsonify({'error': f'Помилка сервера: {str(e)}'}), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    if task_id in progress_data:
        return jsonify(progress_data[task_id])
    return jsonify({'status': 'not_found', 'progress': 0, 'message': 'Задачу не знайдено'}), 404

@app.route('/health')
def health():
    """Перевірка здоров'я сервера"""
    return jsonify({'status': 'ok', 'models_loaded': list(models_cache.keys())})

if __name__ == '__main__':
    try:
        import numpy as np
        logger.info(f"NumPy версія: {np.__version__}")
    except ImportError:
        logger.error("NumPy не встановлено! Встановіть: pip install numpy")
        sys.exit(1)
    
    try:
        import librosa
        logger.info(f"librosa версія: {librosa.__version__} (для точного визначення меж мовлення)")
    except ImportError:
        logger.warning("librosa не встановлено. Буде використано базовий алгоритм.")
        logger.warning("Для кращих результатів встановіть: pip install librosa")
    
    try:
        logger.info(f"OpenAI Whisper завантажено успішно")
    except Exception as e:
        logger.error(f"Помилка з Whisper: {e}")
        sys.exit(1)
    
    logger.info("Сервер запускається на http://0.0.0.0:5000")
    logger.info("Доступ з телефону: http://<IP-вашого-комп'ютера>:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
