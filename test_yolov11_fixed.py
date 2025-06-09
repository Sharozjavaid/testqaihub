#!/usr/bin/env python3
"""
YOLOv11 test script with corrected output parsing
Properly handles the [1, 4, 8400] bbox and [1, 80, 8400] class tensors
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import qai_hub_models
try:
    from qai_hub_models.models.yolov11_det import Model
    import qai_hub as hub
    print("✅ qai_hub_models imported successfully!")
except ImportError as e:
    print(f"❌ Error importing qai_hub_models: {e}")
    sys.exit(1)

def load_and_preprocess_image(image_path, target_size=(640, 640)):
    """Load and preprocess image for YOLOv11"""
    print(f"📸 Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None, None
    
    original_image = image.copy()
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
    
    print(f"✅ Image preprocessed: {image_tensor.shape}")
    return image_tensor, original_image

def extract_detections_corrected(output, confidence_threshold=0.25):
    """Extract detections from YOLOv11 output with correct tensor interpretation"""
    print("🔍 Extracting detections from YOLOv11 output...")
    
    boxes = []
    scores = []
    classes = []
    
    # YOLOv11 output structure: tuple with 2 elements
    # Element 0: tuple with (bbox_tensor, class_tensor)
    # Element 1: list of feature maps (not needed for final detections)
    
    if isinstance(output, (list, tuple)) and len(output) >= 1:
        detection_data = output[0]  # Get the first element
        
        if isinstance(detection_data, (list, tuple)) and len(detection_data) >= 2:
            bbox_tensor = detection_data[0]  # Shape: [1, 4, 8400]
            class_tensor = detection_data[1]  # Shape: [1, 80, 8400]
            
            print(f"Bbox tensor shape: {bbox_tensor.shape}")
            print(f"Class tensor shape: {class_tensor.shape}")
            
            # Remove batch dimension
            bbox_tensor = bbox_tensor.squeeze(0)  # Shape: [4, 8400]
            class_tensor = class_tensor.squeeze(0)  # Shape: [80, 8400]
            
            # Transpose to get [8400, 4] and [8400, 80]
            bbox_tensor = bbox_tensor.transpose(0, 1)  # Shape: [8400, 4]
            class_tensor = class_tensor.transpose(0, 1)  # Shape: [8400, 80]
            
            print(f"After transpose - Bbox: {bbox_tensor.shape}, Class: {class_tensor.shape}")
            
            num_detections = bbox_tensor.shape[0]
            print(f"Processing {num_detections} potential detections")
            
            for i in range(num_detections):
                # Get bounding box coordinates (center format)
                x_center, y_center, width, height = bbox_tensor[i]
                
                # Get class confidences
                class_confidences = class_tensor[i]
                
                # Find the class with highest confidence
                max_conf, class_id = torch.max(class_confidences, dim=0)
                
                if max_conf > confidence_threshold:
                    # Convert from center format to corner format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                    scores.append(max_conf.item())
                    classes.append(class_id.item())
    
    print(f"Extracted {len(boxes)} detections above threshold {confidence_threshold}")
    return boxes, scores, classes

def apply_nms(boxes, scores, classes, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if not boxes:
        return boxes, scores, classes
    
    # Convert to tensors
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    
    # Apply NMS
    keep_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)
    
    # Filter results
    filtered_boxes = [boxes[i] for i in keep_indices]
    filtered_scores = [scores[i] for i in keep_indices]
    filtered_classes = [classes[i] for i in keep_indices]
    
    print(f"After NMS: {len(filtered_boxes)} detections (removed {len(boxes) - len(filtered_boxes)})")
    return filtered_boxes, filtered_scores, filtered_classes

def draw_detections(image, boxes, scores, classes, confidence_threshold=0.25):
    """Draw detection boxes on image"""
    if not boxes:
        print("No detections to draw")
        return image
    
    result_image = image.copy()
    h, w = result_image.shape[:2]
    
    # COCO class names
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        if score < confidence_threshold:
            continue
            
        x1, y1, x2, y2 = box
        
        # Scale coordinates to image size (model input was 640x640)
        x1 = int(x1 * w / 640)
        y1 = int(y1 * h / 640)
        x2 = int(x2 * w / 640)
        y2 = int(y2 * h / 640)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
        
        # Add label
        class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"Class {class_id}"
        label = f"{class_name}: {score:.2f}"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
        cv2.putText(result_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    print(f"Drew {len(boxes)} detection boxes")
    return result_image

def visualize_results(original_image, boxes, scores, classes, output_path):
    """Visualize detection results"""
    print(f"🎨 Visualizing results...")
    
    # Draw detections on image
    result_image = draw_detections(original_image.copy(), boxes, scores, classes)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f"YOLOv11 Detection Results - {len(boxes)} detections")
    plt.axis('off')
    
    # Save result
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Results saved to: {output_path}")

def test_yolov11_corrected():
    """Test YOLOv11 with corrected output parsing"""
    print("🚀 Starting YOLOv11 corrected test...")
    
    # Load the model
    print("📦 Loading YOLOv11 model...")
    try:
        model_wrapper = Model.from_pretrained()
        model = model_wrapper.model
        model.eval()
        print("✅ YOLOv11 model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Find images to test
    images_dirs = [Path("Images"), Path("data/images")]
    image_files = []
    
    for images_dir in images_dirs:
        if images_dir.exists():
            files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
            image_files.extend(files)
            print(f"📂 Found {len(files)} image(s) in {images_dir}")
    
    if not image_files:
        print("❌ No image files found!")
        return False
    
    print(f"📂 Total images found: {len(image_files)}")
    
    # Create results directory
    results_dir = Path("yolov11_corrected_results")
    results_dir.mkdir(exist_ok=True)
    
    # Test on real images
    success_count = 0
    total_detections = 0
    
    for i, image_path in enumerate(image_files[:5]):  # Test all 5 images
        print(f"\n🔍 Testing image {i+1}/{min(5, len(image_files))}: {image_path.name}")
        
        # Load and preprocess image
        image_tensor, original_image = load_and_preprocess_image(image_path)
        if image_tensor is None:
            continue
        
        # Run inference
        try:
            print("🤖 Running YOLOv11 inference...")
            with torch.no_grad():
                output = model(image_tensor)
            
            # Process output with corrected parsing
            boxes, scores, classes = extract_detections_corrected(output, confidence_threshold=0.3)
            
            # Apply NMS to remove duplicates
            if boxes:
                boxes, scores, classes = apply_nms(boxes, scores, classes, iou_threshold=0.5)
            
            print(f"✅ Inference successful! Found {len(boxes)} detections")
            success_count += 1
            total_detections += len(boxes)
            
            # Print detection details
            if boxes:
                print("Detected objects:")
                coco_classes = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                    'hair drier', 'toothbrush'
                ]
                
                for j, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                    class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"Class {class_id}"
                    print(f"  {j+1}. {class_name}: {score:.3f}")
            
            # Visualize results
            output_path = results_dir / f"corrected_{image_path.stem}.png"
            visualize_results(original_image, boxes, scores, classes, output_path)
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if success_count > 0:
        print(f"\n🎉 Testing completed! Successfully processed {success_count} images")
        print(f"📊 Total detections found: {total_detections}")
        print(f"📁 Results saved in: {results_dir}")
        return True
    else:
        print("\n❌ No images were successfully processed")
        return False

def main():
    """Main function"""
    print("🔥 YOLOv11 Corrected Test")
    print("="*40)
    
    # Test with corrected parsing
    success = test_yolov11_corrected()
    
    if success:
        print("\n✅ Tests completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Check the results in yolov11_corrected_results/ folder")
        print("2. Configure API token: qai-hub configure --api_token YOUR_TOKEN")
        print("3. Export for Snapdragon X Elite:")
        print("   python -m qai_hub_models.models.yolov11_det.export --device 'Snapdragon X Elite CRD' --target-runtime onnx")
    else:
        print("\n❌ Tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 