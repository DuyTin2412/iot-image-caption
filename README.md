# iot-image-caption

### 1. Tải và cài đặt Ollama

Truy cập trang chủ [ollama.com](https://ollama.com) và tải bản cài đặt phù hợp.

### 2. Tải model LLaVA

Sau khi cài đặt xong, mở Terminal (hoặc CMD/PowerShell) và chạy lệnh sau để tải model nhận diện ảnh:

```bash
ollama pull llava:7b

```

Chạy lệnh sau để đảm bảo Ollama đang hoạt động

```bash
ollama list
```

### 3. Cài đặt thư viện Python

```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị Model Giọng nói

Tạo folder models và tải hai thư viện này về: https://huggingface.co/rhasspy/piper-voices/tree/main/vi/vi_VN/vais1000/medium
models/vi_VN-vais1000-medium.onnx
models/vi_VN-vais1000-medium.onnx.json

### 5. Cấu hình biến môi trường (.env)

Tạo file .env tại thư mục gốc và điền API Key của Google Gemini vào:
