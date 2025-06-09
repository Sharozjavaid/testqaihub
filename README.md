# YOLOv11 Qualcomm AI Hub Test

A simple test implementation for YOLOv11 object detection using Qualcomm AI Hub, optimized for Snapdragon X Elite processors.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (AMDx64 64-bit for Windows)
- Qualcomm AI Hub account and API token

### Installation
```bash
# Clone repository
git clone https://github.com/Sharozjavaid/testqaihub.git
cd testqaihub

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure Qualcomm AI Hub
qai-hub configure --api_token YOUR_API_TOKEN
```

### Usage
```bash
# Run YOLOv11 test
python test_yolov11_fixed.py
```

### Deploy to Snapdragon X Elite
```bash
# Export model for Snapdragon X Elite
python -m qai_hub_models.models.yolov11_det.export \
    --device "Snapdragon X Elite CRD" \
    --target-runtime onnx
```

## 📁 Files Included
- `test_yolov11_fixed.py` - Main test script with corrected output parsing
- `requirements.txt` - Python dependencies
- `data/images/` - Sample test images

## 🎯 Features
- ✅ Corrected YOLOv11 output tensor parsing
- ✅ Proper bbox and class confidence handling
- ✅ Non-Maximum Suppression (NMS)
- ✅ Visualization with detection boxes
- ✅ Support for Snapdragon X Elite deployment

## 📊 Expected Performance on Snapdragon X Elite
- Model Size: ~2.83 MB (INT8 quantized)
- Input Resolution: 640x640
- Target Latency: <50ms
- Target Throughput: >20 FPS

## 🔧 Troubleshooting

### Python Architecture
Ensure you're using AMDx64 Python on Windows:
```bash
python -c "import platform; print(platform.architecture())"
# Should show: ('64bit', 'WindowsPE')
```

### Device Detection
Verify your Snapdragon device is recognized:
```bash
qai-hub devices
```

## 📝 Getting API Token
1. Go to [Qualcomm AI Hub](https://aihub.qualcomm.com/)
2. Create account/login
3. Navigate to your profile/settings to get API token
4. Configure: `qai-hub configure --api_token YOUR_TOKEN` 