import os
import warnings
import time
import wave
import base64
import io
import json
import datetime
import uuid
import torch
import re  # <--- THÊM CÁI NÀY
from flask import Flask, request, jsonify
from PIL import Image
import ollama
from transformers import BlipProcessor, BlipForConditionalGeneration
from google import genai
from dotenv import load_dotenv
from piper import PiperVoice

warnings.filterwarnings("ignore")
load_dotenv()

app = Flask(__name__)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir 

LOG_DIR = os.path.join(project_root, 'logs')
IMG_LOG_DIR = os.path.join(LOG_DIR, 'images')
AUDIO_LOG_DIR = os.path.join(LOG_DIR, 'audio')
HISTORY_FILE = os.path.join(LOG_DIR, 'history.jsonl')

os.makedirs(IMG_LOG_DIR, exist_ok=True)
os.makedirs(AUDIO_LOG_DIR, exist_ok=True)

# --- CẤU HÌNH MODEL ---
OLLAMA_MODEL = "qwen3-vl:8b" 
GEMINI_MODEL = "gemini-2.5-flash" # Lưu ý: Đã sửa lại thành 1.5-flash (2.5 chưa chạy được public)
MODEL_TTS_FILENAME = "vi_VN-vais1000-medium.onnx"
MODEL_TTS_PATH = os.path.join(project_root, 'models', MODEL_TTS_FILENAME)

genai_client = None
piper_voice = None
blip_processor = None
blip_model = None

# --- HÀM LÀM SẠCH VĂN BẢN (MỚI) ---
def clean_text_for_audio(text):
    if not text:
        return ""
    
    # 1. Xóa các ký tự markdown của AI (như **bold**, *italic*, # header)
    text = re.sub(r'[\*#_`]', '', text)
    
    # 2. Xóa các dấu ngoặc kép/đơn hoặc ngoặc lạ để tránh đọc "mở ngoặc/đóng ngoặc"
    text = re.sub(r'[\"\'\(\)\[\]\{\}]', '', text)
    
    # 3. Xóa các ký tự đặc biệt khác nhưng giữ lại dấu câu cơ bản (. , ? !) và tiếng Việt
    # Nếu muốn giữ lại % hoặc số thì thêm vào regex, hiện tại giữ nguyên để Piper tự xử lý số
    
    # 4. Xóa khoảng trắng thừa (ví dụ 2 dấu cách liền nhau)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def save_log_entry(entry):
    try:
        with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error saving log: {e}")

def get_genai_client():
    global genai_client
    if genai_client:
        return genai_client

    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Missing GEMINI_API_KEY")
        return None

    try:
        genai_client = genai.Client(api_key=api_key)
        return genai_client
    except Exception as e:
        print(f"❌ GenAI Client Init Error: {e}")
        return None

def load_piper_voice():
    global piper_voice
    if piper_voice is not None:
        return piper_voice

    if os.path.exists(MODEL_TTS_PATH):
        try:
            piper_voice = PiperVoice.load(MODEL_TTS_PATH)
            print(f"✅ Piper voice loaded: {MODEL_TTS_FILENAME}")
        except Exception as e:
            print(f"❌ Piper load error: {e}")
    else:
        print(f"❌ Piper model not found at: {MODEL_TTS_PATH}")
    return piper_voice

def synthesize_speech(text):
    voice = load_piper_voice()
    if not voice:
        return None
    try:
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None

def setup_blip():
    global blip_processor, blip_model
    if blip_model is not None:
        return

    try:
        print("⏳ Loading BLIP model...")
        repo_id = "Salesforce/blip-image-captioning-base"
        blip_processor = BlipProcessor.from_pretrained(repo_id)
        blip_model = BlipForConditionalGeneration.from_pretrained(repo_id)
        blip_model.eval()
        try:
            blip_model = torch.quantization.quantize_dynamic(
                blip_model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except:
            pass
        print("✅ BLIP loaded.")
    except Exception as e:
        print(f"❌ BLIP load error: {e}")

def generate_short_caption(pil_image):
    setup_blip()
    if not blip_model:
        return "BLIP not loaded"
    try:
        raw_image = pil_image.convert('RGB').resize((384, 384), Image.Resampling.LANCZOS)
        inputs = blip_processor(raw_image, return_tensors="pt")
        with torch.no_grad():
            outputs = blip_model.generate(
                **inputs, max_length=50, num_beams=2, early_stopping=True
            )
        return blip_processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"❌ BLIP error: {e}")
        return "An image."

def generate_long_caption(image_bytes):
    try:
        system_prompt = (
            "Role: Visual Assistant for a blind person. "
            "Task: Describe the scene objectively. "
            "Structure your response as follows: "
            "1. Start exactly with 'In front of you is...'. "
            "2. Mention the main object and its position (center, left, right). "
            "3. Identify any immediate obstacles or safety hazards. "
            "4. Read any visible text accurately. "
            "Keep it concise (3-4 sentences)."
        )
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': system_prompt,
                'images': [image_bytes]
            }]
        )
        return response['message']['content']
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return "Error analyzing image."

# --- API ENDPOINTS ---

@app.route('/caption', methods=['POST'])
def caption_api():
    try:
        start_total = time.time()
        
        # 1. Nhận ảnh
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        image_bytes = file.read()
        mode = request.form.get('mode', 'long').lower() 

        request_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        img_path = os.path.join(IMG_LOG_DIR, f"{request_id}.jpg")
        with open(img_path, 'wb') as f:
            f.write(image_bytes)

        # 2. Tạo caption tiếng Anh
        start_cap = time.time()
        if mode == 'short':
            pil_image = Image.open(io.BytesIO(image_bytes))
            caption_en = generate_short_caption(pil_image)
        else:
            caption_en = generate_long_caption(image_bytes)
        time_cap = time.time() - start_cap

        # 3. Dịch sang tiếng Việt
        start_trans = time.time()
        caption_vi = None
        try:
            client = get_genai_client()
            if client:
                prompt = (
                    "Dịch mô tả hình ảnh sau sang tiếng Việt tự nhiên cho người khiếm thị. "
                    "Chỉ trả về nội dung dịch, không thêm lời dẫn."
                )
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[f"{prompt}\n\n{caption_en}"]
                )
                if hasattr(response, 'text'):
                    caption_vi = response.text.strip()
                else:
                    caption_vi = str(response).strip()
            else:
                caption_vi = "Lỗi kết nối dịch thuật."
        except Exception as e:
            print(f"❌ Translation error: {e}")
            caption_vi = "Không thể dịch mô tả."
        time_trans = time.time() - start_trans

        # 4. Tạo âm thanh (TTS)
        start_audio = time.time()
        audio_base64 = None
        audio_path = None
        
        # Chọn văn bản để đọc
        raw_text_to_speak = caption_vi if caption_vi else caption_en
        
        if raw_text_to_speak:
            # ---> BƯỚC QUAN TRỌNG: LÀM SẠCH VĂN BẢN TRƯỚC KHI ĐỌC <---
            clean_text = clean_text_for_audio(raw_text_to_speak)
            print(f"🔹 Text for TTS (Cleaned): {clean_text}") # In ra để kiểm tra
            
            audio_data = synthesize_speech(clean_text)
            if audio_data:
                audio_path = os.path.join(AUDIO_LOG_DIR, f"{request_id}_vi.wav")
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        time_audio = time.time() - start_audio

        total_time = time.time() - start_total

        # 5. Lưu Log
        save_log_entry({
            "id": request_id,
            "mode": mode,
            "original_en": caption_en,
            "translated_vi": caption_vi,
            "image_path": img_path,
            "audio_path": audio_path,
            "timestamp": datetime.datetime.now().isoformat()
        })

        # 6. Trả về kết quả
        response_data = {
            "id": request_id,
            "mode": mode,
            "original_caption": caption_en,
            "translated_caption": caption_vi,
            "processing_time": {
                "captioning": round(time_cap, 2),
                "translation": round(time_trans, 2),
                "tts": round(time_audio, 2),
                "total": round(total_time, 2)
            }
        }

        if audio_base64:
            response_data["audio"] = {
                "data": audio_base64,
                "format": "wav",
                "encoding": "base64"
            }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "active",
        "mode_short_ready": blip_model is not None,
        "mode_long_ready": True,
        "tts_ready": piper_voice is not None,
        "gemini_model": GEMINI_MODEL
    })

if __name__ == '__main__':
    print("🚀 Starting Flask API...")
    load_piper_voice()
    setup_blip()
    app.run(host='0.0.0.0', port=5000, debug=False)