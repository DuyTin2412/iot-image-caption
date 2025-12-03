import os
import warnings
import time
import wave
import base64
from flask import Flask, request, jsonify, send_file
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from google import genai
from dotenv import load_dotenv
from piper import PiperVoice
import io
import torch
import torch.quantization

warnings.filterwarnings("ignore")

load_dotenv()

app = Flask(__name__)

llm_processor = None
llm_model = None
genai_client = None
piper_voice = None


def quantize_model(model):
    try:
        print("Applying dynamic quantization to model...")
        start_time = time.time()
        
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        quantization_time = time.time() - start_time
        print(f"Model quantization completed in {quantization_time:.2f} seconds")
        
        # Get model size comparison
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
        
        print(f"Model size reduced from {original_size:.1f}MB to {quantized_size:.1f}MB "
              f"({((original_size - quantized_size) / original_size * 100):.1f}% reduction)")
        
        return quantized_model
        
    except Exception as e:
        print(f"Quantization failed: {e}. Using original model.")
        return model


def load_model():
    global llm_processor, llm_model
    
    enable_quantization = os.environ.get('ENABLE_QUANTIZATION', 'true').lower() == 'true'

    preferred_local_dir = os.environ.get('BLIP_MODEL_DIR') or os.path.join(
        os.path.dirname(__file__),
        'models',
        'blip-image-captioning-base'
    )

    model_source_desc = None
    print("Loading BLIP model... This may take a moment.")

    # Try local first if the directory exists and looks valid
    if os.path.isdir(preferred_local_dir):
        try:
            llm_processor = BlipProcessor.from_pretrained(preferred_local_dir, local_files_only=True)
            llm_model = BlipForConditionalGeneration.from_pretrained(preferred_local_dir, local_files_only=True)
            model_source_desc = f"local path: {preferred_local_dir}"
        except Exception as local_err:
            print(f"Failed loading model from local path '{preferred_local_dir}'. Will try online. Error: {local_err}")

    # If local load didn't happen, fall back to hub
    if llm_processor is None or llm_model is None:
        repo_id = "Salesforce/blip-image-captioning-base"
        llm_processor = BlipProcessor.from_pretrained(repo_id)
        llm_model = BlipForConditionalGeneration.from_pretrained(repo_id)
        model_source_desc = f"Hugging Face Hub repo: {repo_id}"

    # Set model to evaluation mode
    llm_model.eval()
    
    # Apply quantization if enabled
    if enable_quantization:
        llm_model = quantize_model(llm_model)
        quantization_status = " (quantized for faster CPU inference)"
    else:
        quantization_status = " (full precision)"
    
    print(f"Model loaded successfully and cached in memory from {model_source_desc}{quantization_status}!")


def load_piper_voice():
    """Load Piper TTS voice model."""
    global piper_voice
    if piper_voice is not None:
        return piper_voice

    # Check for custom voice path in environment
    voice_path = os.environ.get('PIPER_VOICE_PATH')

    if voice_path and os.path.exists(voice_path):
        print(f"Loading Piper voice from: {voice_path}")
        piper_voice = PiperVoice.load(voice_path)
    else:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'models', 'vi_VN-vais1000-medium.onnx'),
            './models/vi_VN-vais1000-medium.onnx',
            './vi_VN-vais1000-medium.onnx'
        ]

        voice_found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading Piper voice from: {path}")
                piper_voice = PiperVoice.load(path)
                voice_found = True
                break

    print("Piper voice loaded successfully!")
    return piper_voice


def get_genai_client():
    global genai_client
    if genai_client is not None:
        return genai_client
    api_key = (
            os.environ.get('GEMINI_API_KEY') or
            os.environ.get('GOOGLE_API_KEY')
    )
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment/.env")
    genai_client = genai.Client(api_key=api_key)
    return genai_client


def synthesize_speech(text):
    voice = load_piper_voice()
    if voice is None:
        return None

    try:
        # Create in-memory WAV file
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return None


def analyse_image(image):
    global llm_processor, llm_model
    try:
        start_time = time.time()
        
        # Preprocessing optimizations for faster inference
        raw_image = image.convert('RGB')
        target_size = (384, 384)  # Optimized size for BLIP
        raw_image = raw_image.resize(target_size, Image.Resampling.LANCZOS)
        
        preprocessing_time = time.time() - start_time
        
        # Process with model
        inputs = llm_processor(raw_image, return_tensors="pt")
        
        # Optimized generation parameters for speed on CPU/quantized model
        generation_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():  # Disable gradients for faster inference
                outputs = llm_model.generate(
                    **inputs,
                    max_length=25,        # Reduced for faster generation
                    num_beams=2,          # Reduced from 3 for quantized model
                    early_stopping=True,  
                    do_sample=False
                )
        
        generation_time = time.time() - generation_start
        
        caption = llm_processor.decode(outputs[0], skip_special_tokens=True)
        
        total_time = time.time() - start_time
        print(f"Image analysis completed in {total_time:.3f}s "
              f"(preprocessing: {preprocessing_time:.3f}s, generation: {generation_time:.3f}s)")
        
        return caption

    except Exception as e:
        print(f"An error occurred during image analysis: {e}")
        import traceback
        traceback.print_exc()
        return "Error processing the image."


@app.route('/translate', methods=['POST'])
def translate_with_gemini():
    try:
        text = request.form.get('text')
        if not text:
            return jsonify({'error': 'Missing form field: text'}), 400
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({'error': 'Invalid file format. Only PNG, JPG, and JPEG are supported'}), 400

        image_bytes = file.read()
        mime_type = 'image/png' if file_extension == 'png' else 'image/jpeg'
        client = get_genai_client()
        contents = [
            {'mime_type': mime_type, 'data': image_bytes},
            {
                'text': (
                    'Translate the following text to Vietnamese. '
                    'Use the provided image as contextual reference to resolve ambiguities.\n\n'
                    f'Text: {text}'
                )
            }
        ]

        start_time = time.time()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        processing_time = time.time() - start_time

        translated = getattr(response, 'text', None) or getattr(response, 'output_text', None)
        if not translated:
            translated = str(response)

        return jsonify({
            'translation': translated,
            'model': 'gemini-2.5-flash-lite',
            'processing_time_seconds': round(processing_time, 2)
        })
    except RuntimeError as cfg_err:
        return jsonify({'error': str(cfg_err)}), 500
    except Exception as e:
        print(f"Error in translate_with_gemini endpoint: {e}")
        return jsonify({'error': 'Internal server error occurred while translating'}), 500


@app.route('/caption', methods=['POST'])
def caption_image():
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''

        if file_extension not in allowed_extensions:
            return jsonify({'error': 'Invalid file format. Only PNG, JPG, and JPEG are supported'}), 400

        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        total_start = time.time()

        # Captioning phase
        cap_start = time.time()
        caption = analyse_image(image)
        captioning_time = time.time() - cap_start

        # Translating phase (simple text translation only, no image)
        trans_start = time.time()
        try:
            client = get_genai_client()
            contents = [
                f"Translate this text to Vietnamese. Respond with ONLY the translated sentence, nothing else:\n{caption}"
            ]
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=contents
            )
            translated = getattr(response, 'text', None) or getattr(response, 'output_text', None) or str(response)
        except Exception as translate_err:
            print(f"Translation failed: {translate_err}")
            translated = None
        translating_time = time.time() - trans_start

        # Audio synthesis phase (optional)
        audio_start = time.time()
        audio_data = None
        audio_base64 = None

        # Check if audio is requested (default to true for auto-include)
        include_audio = request.form.get('include_audio', 'true').lower() == 'true'
        audio_language = request.form.get('audio_language', 'translated').lower()  # 'original' or 'translated'

        if include_audio:
            # Choose which text to synthesize
            text_to_synthesize = translated if audio_language == 'translated' and translated else caption
            audio_data = synthesize_speech(text_to_synthesize)
            if audio_data:
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        audio_time = time.time() - audio_start
        total_time = time.time() - total_start

        print(
            f"Captioning: {captioning_time:.2f}s, Translating: {translating_time:.2f}s, Audio: {audio_time:.2f}s, Total: {total_time:.2f}s")

        # Return response with both original and translated caption, audio, and timings
        response_data = {
            'original_caption': caption,
            'translated_caption': translated,
            'timing': {
                'captioning_seconds': round(captioning_time, 2),
                'translating_seconds': round(translating_time, 2),
                'audio_seconds': round(audio_time, 2),
                'total_seconds': round(total_time, 2)
            }
        }

        if include_audio and audio_base64:
            response_data['audio'] = {
                'data': audio_base64,
                'format': 'wav',
                'encoding': 'base64',
                'text_used': text_to_synthesize,
                'language': audio_language
            }
        elif include_audio:
            response_data['audio'] = {
                'error': 'Audio synthesis failed or voice model not available'
            }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in caption_image endpoint: {e}")
        return jsonify({'error': 'Internal server error occurred while processing image'}), 500


@app.route('/synthesize', methods=['POST'])
def synthesize_text():
    """Convert text to speech using Piper TTS"""
    try:
        # Get text from request
        text = request.form.get('text') or request.json.get('text') if request.is_json else None

        if not text:
            return jsonify({'error': 'Missing text parameter'}), 400

        if len(text.strip()) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Check if text is too long (optional limit)
        max_length = int(os.environ.get('MAX_TTS_LENGTH', 1000))
        if len(text) > max_length:
            return jsonify({'error': f'Text too long. Maximum {max_length} characters allowed'}), 400

        start_time = time.time()
        audio_data = synthesize_speech(text)
        processing_time = time.time() - start_time

        if audio_data is None:
            return jsonify({'error': 'Audio synthesis failed or voice model not available'}), 500

        # Check if client wants raw audio file or base64 encoded data
        return_format = request.form.get('format', 'json').lower()

        if return_format == 'audio':
            # Return raw audio file
            audio_io = io.BytesIO(audio_data)
            audio_io.seek(0)
            return send_file(
                audio_io,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='synthesized_speech.wav'
            )
        else:
            # Return JSON with base64 encoded audio
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            return jsonify({
                'audio': {
                    'data': audio_base64,
                    'format': 'wav',
                    'encoding': 'base64',
                    'text_used': text
                },
                'processing_time_seconds': round(processing_time, 2)
            })

    except Exception as e:
        print(f"Error in synthesize_text endpoint: {e}")
        return jsonify({'error': 'Internal server error occurred during speech synthesis'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Image Captioning API is running'})


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Image Captioning API using BLIP model + Gemini translation + Piper TTS',
        'endpoints': {
            '/caption': 'POST - Upload image for captioning (with optional audio synthesis)',
            '/translate': 'POST - Translate text to Vietnamese with image context',
            '/synthesize': 'POST - Convert text to speech using Piper TTS',
            '/health': 'GET - Health check',
            '/': 'GET - API information'
        },
        'usage': {
            '/caption': 'POST multipart/form-data: image=<file>, include_audio=<true/false> [default: true], audio_language=<original/translated> [default: translated]',
            '/translate': 'POST multipart/form-data: text=<string>, image=<file>. Requires env GEMINI_API_KEY or GOOGLE_API_KEY',
            '/synthesize': 'POST form-data or JSON: text=<string>, format=<json/audio>. Requires Piper voice model (set PIPER_VOICE_PATH)'
        },
        'audio_features': {
            'tts_engine': 'Piper TTS',
            'supported_formats': ['wav'],
            'output_options': ['base64 JSON', 'raw audio file'],
            'voice_models': {
                'vietnamese': 'vi_VN-vais1000-medium (recommended for Vietnamese text)',
                'english': 'en_US-lessac-medium (for English text)'
            },
            'setup_required': 'Place voice model (.onnx file) in models/ folder or set PIPER_VOICE_PATH'
        }
    })


if __name__ == '__main__':
    print("Starting Image Captioning API...")
    load_model()
    load_piper_voice()  # Load TTS voice model

    print("API will be available at: http://localhost:5500")
    print("Endpoints:")
    print("  POST /caption - Upload images for captioning (with optional audio)")
    print("  POST /translate - Translate text with image context")
    print("  POST /synthesize - Convert text to speech")
    print("  GET /health - Health check")
    print("  GET / - API information")

    app.run(host='0.0.0.0', port=5500, debug=False)
