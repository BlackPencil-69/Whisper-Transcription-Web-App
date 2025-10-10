import os
import sys
import time
import whisper
import threading
import subprocess
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path

# ---------- Конфігурація ----------
app = Flask(__name__, template_folder="templates")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mkv', 'flac', 'm4a', 'ogg', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

progress_data = {}
models_cache = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Допоміжні функції ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio(video_path, audio_path):
    """Витягує аудіо з відео через FFmpeg"""
    try:
        if not os.path.exists(video_path):
            logger.error(f"Відео не існує: {video_path}")
            return False

        if os.path.exists(audio_path):
            os.remove(audio_path)

        result = subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_path, '-y'
        ], check=True, capture_output=True, text=True, timeout=300)

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            logger.error("Помилка: аудіо не створено або занадто мале")
            return False

        logger.info(f"Аудіо створено: {audio_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg помилка: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"extract_audio error: {e}")
        return False


def load_whisper_model(model_size="base"):
    try:
        if model_size not in models_cache:
            logger.info(f"Завантаження моделі Whisper: {model_size}")
            models_cache[model_size] = whisper.load_model(model_size)
        return models_cache[model_size]
    except Exception as e:
        logger.error(f"Помилка завантаження моделі: {e}")
        raise


def format_timestamp(seconds):
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(segments):
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg.get('start', 0))
        end = format_timestamp(seg.get('end', 0))
        text = seg.get('text', '').strip()
        srt_lines += [str(i), f"{start} --> {end}", text, ""]
    return "\n".join(srt_lines)


# ---------- Транскрипція ----------
def transcribe_audio(file_path, language, model_size, task_id):
    srt_path = None
    try:
        model = load_whisper_model(model_size)
        options = {'verbose': False, 'fp16': False, 'task': 'transcribe'}
        if language != 'auto':
            options['language'] = language

        logger.info(f"Транскрипція файлу {file_path}")
        result = model.transcribe(file_path, **options)
        text = result.get("text", "").strip()
        segments = result.get("segments") or []

        if not segments and text:
            segments = [{'start': 0, 'end': 1, 'text': text}]

        srt_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(generate_srt(segments))

        progress_data[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Готово!",
            "result": {
                "full_text": text or "(текст не розпізнано)",
                "srt_filename": f"{task_id}_subtitles.srt",
                "detected_language": result.get("language", "невідома")
            }
        }

    except Exception as e:
        logger.error(f"Помилка транскрипції: {e}", exc_info=True)
        progress_data[task_id] = {"status": "error", "message": str(e)}
        if srt_path and os.path.exists(srt_path):
            os.remove(srt_path)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------- Flask маршрути ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не знайдено'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Файл не обрано'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Непідтримуваний формат'}), 400

        language = request.form.get('language', 'uk')
        model_size = request.form.get('model_size', 'base')
        task_id = str(int(time.time() * 1000))

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(save_path)

        # Якщо відео → конвертуємо
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in {'mp4', 'avi', 'mkv', 'webm'}:
            audio_path = save_path.rsplit('.', 1)[0] + ".wav"
            if not extract_audio(save_path, audio_path):
                return jsonify({'error': 'Помилка конвертації відео'}), 500
            os.remove(save_path)
            save_path = audio_path

        threading.Thread(target=transcribe_audio, args=(save_path, language, model_size, task_id), daemon=True).start()
        return jsonify({'task_id': task_id})

    except Exception as e:
        logger.error(f"upload_file error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/progress/<task_id>')
def progress(task_id):
    return jsonify(progress_data.get(task_id, {'status': 'not_found', 'progress': 0}))


@app.route('/download/<filename>')
def download(filename):
    if not filename.endswith(".srt"):
        return jsonify({'error': 'Недозволений тип файлу'}), 403
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Файл не знайдено'}), 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True, download_name=filename)


# ---------- Запуск ----------
if __name__ == '__main__':
    try:
        import numpy
        logger.info(f"NumPy: {numpy.__version__}")
    except Exception:
        logger.error("Встановіть numpy: pip install numpy")
        sys.exit(1)

    logger.info("Сервер запущено: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
