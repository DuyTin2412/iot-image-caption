import os
import warnings
import time
import wave
import base64
import io
from flask import Flask, request, jsonify
import ollama
from google import genai
from dotenv import load_dotenv
from piper import PiperVoice

warnings.filterwarnings("ignore")
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

load_dotenv(os.path.join(project_root, '.env'))

app = Flask(__name__)
OLLAMA_MODEL = "llava:7b"
MODEL_FILENAME = "vi_VN-vais1000-medium.onnx"
MODEL_PATH = os.path.join(project_root, 'models', MODEL_FILENAME)
genai_client = None
piper_voice = None

def check_ollama_connection():
    try:
        ollama.list()
        print(f"Connected to Ollama successfully. Using model: {OLLAMA_MODEL}")
    except Exception as e:
        print("Warning: Cannot connect to Ollama server.")
        print(f"Error: {e}")

def load_piper_voice():
    global piper_voice
    if piper_voice is not None:
        return piper_voice

    print(f"🔍 Looking for Piper voice at: {MODEL_PATH}")

    if os.path.exists(MODEL_PATH):
        try:
            piper_voice = PiperVoice.load(MODEL_PATH)
            print("Piper voice loaded successfully.")
        except Exception as e:
            print(f"Error loading model file: {e}")
    else:
        print(f"Piper voice not found at {MODEL_PATH}")
        print("Please check if the file name matches exactly.")

    return piper_voice

def get_genai_client():
    global genai_client
    if genai_client is not None:
        return genai_client
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        print("Warning: Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env")
        return None
    
    genai_client = genai.Client(api_key=api_key)
    return genai_client

def synthesize_speech(text):
    voice = load_piper_voice()
    if voice is None:
        return None

    try:
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def analyze_image(image_bytes):
    try:
        start_time = time.time()
        system_prompt = """Role:
You are a helpful visual assistant for people who are blind or have vision impairments.
Task:
Describe the scenery and environment directly in front of the person as if you are their eyes. 
Output:
Begin exactly with: "In front of you is..."
Then continue with 3-4 factual sentences describing objects, obstacles, and people.
Constraints:
- Do NOT greet.
- Do NOT give advice.
- Keep it short and factual."""

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'user', 'content': system_prompt, 'images': [image_bytes]}
            ]
        )
        
        caption = response['message']['content']
        total_time = time.time() - start_time
        print(f"Ollama analysis completed in {total_time:.3f}s")
        return caption

    except Exception as e:
        print(f"Ollama Error: {e}")
        return "Error processing image."

@app.route('/caption', methods=['POST'])
def caption_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        image_bytes = file.read()
        total_start = time.time()

        cap_start = time.time()
        caption = analyze_image(image_bytes)
        captioning_time = time.time() - cap_start

        trans_start = time.time()
        translated = None
        translated_prompt = "Translate the following description into natural Vietnamese for a blind person. Output only the translation:"
        
        try:
            client = get_genai_client()
            if client:
                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=[f"{translated_prompt}\n\n{caption}"]
                )
                translated = getattr(response, 'text', None) or str(response)
                translated = translated.strip()
        except Exception as translate_err:
            print(f"Translation error: {translate_err}")

        translating_time = time.time() - trans_start

        audio_start = time.time()
        audio_base64 = None
        include_audio = request.form.get('include_audio', 'true').lower() == 'true'

        if include_audio:
            text_to_speak = translated if translated else caption
            if text_to_speak:
                audio_data = synthesize_speech(text_to_speak)
                if audio_data:
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        audio_time = time.time() - audio_start
        total_time = time.time() - total_start

        response_data = {
            'original_caption': caption,
            'translated_caption': translated,
            'timing': {
                'captioning': round(captioning_time, 2),
                'translating': round(translating_time, 2),
                'audio': round(audio_time, 2),
                'total': round(total_time, 2)
            }
        }

        if include_audio and audio_base64:
            response_data['audio'] = {
                'data': audio_base64,
                'format': 'wav',
                'encoding': 'base64'
            }

        return jsonify(response_data)

    except Exception as e:
        print(f"/caption endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    voice_status = "Loaded" if piper_voice else "Not Loaded"
    return jsonify({
        'status': 'healthy', 
        'backend': 'Ollama + Gemini + Piper',
        'voice_model': MODEL_FILENAME,
        'voice_status': voice_status
    })

if __name__ == '__main__':
    print(f"🚀 Starting Flask API...")
    print(f"📂 Project Root detected at: {project_root}")
    
    check_ollama_connection()
    load_piper_voice()

    app.run(host='0.0.0.0', port=5000, debug=False)