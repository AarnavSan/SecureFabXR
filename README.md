# Secure Fab XR - Industrial Part Detection in Mixed Reality

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-PICO%204%20Ultra-orange.svg)](https://www.picoxr.com/)
[![OpenXR](https://img.shields.io/badge/OpenXR-1.0-green.svg)](https://www.khronos.org/openxr/)

A mixed reality application for industrial manufacturing that detects and identifies specific parts from a collection of components in the worker's view. Built using SecureMR APIs and YOLOv11 detection model on PICO 4 Ultra devices.

![Demo of part detection in mixed reality](docs/Demo-YOLO.gif)

## 🎯 What Does This Project Do?

**Secure Fab XR** is designed for manufacturing and assembly environments where workers need to identify specific parts from bins, workbenches, or assembly stations containing multiple components. The application:

- **Identifies specific industrial parts** from among multiple objects in the field of view using a YOLOv11 detection model
- **Highlights detected parts** with 3D bounding boxes and labels overlaid directly in the worker's visual field
- **Provides real-time spatial guidance** by mapping 2D camera detections to precise 3D world coordinates
- **Operates entirely on-device** ensuring privacy and minimizing latency for industrial environments
- **Enables custom part recognition** through Docker-based ML model training and conversion pipeline

### Primary Use Cases

🏭 **Parts Picking & Kitting** - Identify correct components from bins containing similar parts  
🔧 **Assembly Verification** - Confirm proper parts are selected before assembly steps  
📦 **Quality Inspection** - Detect and verify parts match specifications during production  
🛠️ **Maintenance & Repair** - Identify replacement parts in complex machinery environments  
📊 **Inventory Management** - Real-time tracking of parts presence and location on workstations

### Key Features

✅ **Multi-object detection** - Identifies target parts even when surrounded by other components  
✅ **Real-time inference** - Hardware-accelerated on-device processing for interactive frame rates  
✅ **Privacy-first architecture** - All data processing happens locally on the headset  
✅ **Stereo depth mapping** - Accurate 3D positioning using dual camera stereo vision  
✅ **Custom model training** - Docker pipeline for training models on your specific parts  
✅ **Production-ready utilities** - Reusable SecureMR components for building industrial XR apps

---

## 🏗️ Architecture Overview

The application implements a **multi-stage pipeline architecture** optimized for real-time industrial part detection:

```
┌──────────────────┐      ┌───────────────────┐      ┌──────────────────┐      ┌──────────────┐
│  VST Camera      │ ───> │ YOLOv11 Model     │ ───> │ 2D-to-3D         │ ───> │  3D Overlay  │
│  Image Pipeline  │      │ Inference + NMS   │      │  Spatial Mapping │      │  Rendering   │
└──────────────────┘      └───────────────────┘      └──────────────────┘      └──────────────┘
     Camera Feed             Part Detection            Depth Estimation         Visual Feedback
```

### Pipeline Components

1. **VST Image Pipeline** 
   - Captures stereo camera feed from PICO 4 Ultra headset
   - Preprocesses images (format conversion, normalization) for ML inference
   - Manages camera timestamps and intrinsic/extrinsic matrices

2. **Model Inference Pipeline**
   - Runs YOLOv11 object detection model on preprocessed images
   - Performs class filtering to identify only target industrial parts
   - Applies Non-Maximum Suppression (NMS) to eliminate duplicate detections
   - Outputs bounding boxes with confidence scores and class labels

3. **2D-to-3D Mapping Pipeline**
   - Transforms 2D bounding boxes into 3D spatial coordinates
   - Utilizes stereo vision for depth estimation
   - Generates world-space positions for detected parts relative to the headset

4. **3D Rendering Pipeline**
   - Visualizes detections as 3D overlays anchored to physical parts
   - Renders bounding boxes and text labels using GLTF mesh assets
   - Updates overlay positions and orientations in real-time as the user moves

All pipelines execute asynchronously with synchronized data flow through global tensors, ensuring low latency and high throughput for interactive manufacturing applications.

---

## 📦 What's Included / Not Included

### ✅ Included in This Repository

**Core Application**
- Complete C++ source code for industrial part detection sample app
- OpenXR 1.0 integration for XR session management
- SecureMR framework integration for on-device AI pipelines
- Multi-threaded pipeline execution with synchronized tensor operations

**ML Model Infrastructure**
- YOLOv11 detection model configuration and integration
- Model inference pipeline with hardware acceleration support
- Post-processing operators (NMS, class filtering, confidence thresholding)
- Tensor management utilities for efficient memory usage

**Development & Deployment Tools**
- Docker-based ML model conversion pipeline (ONNX → SecureMR format)
- Vulkan shader programs for hardware-accelerated rendering
- Gradle build configuration for Android deployment
- Utility classes for OpenXR and SecureMR API simplification

**Visual Assets & Resources**
- GLTF 3D models for bounding box and label visualization
- Sample model configuration files
- Demo assets for testing and validation

**Documentation & Examples**
- Architectural documentation and code walkthroughs
- API usage examples and best practices
- Build, deployment, and troubleshooting guides

### ⚠️ Not Included

**Hardware & Software Prerequisites**
- PICO 4 Ultra headset (required to run application)
- Android Studio IDE with SDK/NDK (must install separately per [PICO documentation](https://developer-cn.picoxr.com/document/native/securemr-overview/))
- Docker Desktop (optional, for custom model training)

**Pre-trained Models**
- Industrial part detection models (demo uses generic objects; train your own for specific parts)
- Custom trained weights for your manufacturing environment

**Production Features**
- Enterprise safety and compliance features
- Multi-user collaboration or cloud synchronization
- Production database integration or ERP connectivity
- Advanced analytics or logging infrastructure

**Important**: This is a development sample and demonstration of SecureMR capabilities. Production deployments require additional safety validation, testing, and compliance verification for your specific industrial environment.

---

## 🚀 Installation & Setup

Follow these instructions to build and run Secure Fab XR on your PICO 4 Ultra device.

### Prerequisites

#### (A) Hardware Requirements
- **PICO 4 Ultra** headset with OS version **≥ 5.14.0U** (latest system update required)
- **USB-C cable** for connecting headset to development computer

#### (B) Software Requirements

1. **Android Studio** (latest stable version recommended)
   - Download from: https://developer.android.com/studio
   
2. **Android SDK Platform 34**
   - Install via Android Studio SDK Manager
   
3. **Android NDK version 25.x.x** (recommended)
   - Install via Android Studio SDK Manager
   
4. **Gradle 8.7** and **Android Gradle Plugin 8.3.2**
   - Typically bundled with Android Studio
   
5. **Java JDK 17 or higher** (Java 21 recommended)
   - Required by Android Gradle Plugin
   
6. **Docker Desktop** (optional)
   - Only needed for custom ML model conversion
   - Download from: https://www.docker.com/products/docker-desktop

---

### Step-by-Step Installation

#### 1️⃣ Configure Android Studio and SDK

1. **Install Android Studio** and launch it

2. **Open SDK Manager**:
   - Go to `Tools` → `SDK Manager` in Android Studio menu
   
3. **Install SDK Platform**:
   - Navigate to the `SDK Platforms` tab
   - Check the box for **Android 14.0 (API Level 34)**
   - Click `Apply` to install
   
4. **Install NDK**:
   - Navigate to the `SDK Tools` tab
   - Check `Show Package Details` at the bottom right
   - Expand `NDK (Side by side)`
   - Select **version 25.x.x** (e.g., 25.2.9519653)
   - Click `Apply` to install
   
5. **Verify Installation**:
   - NDK will be installed to: `<SDK Location>/ndk/25.x.x`
   - You can find your SDK location in the SDK Manager window

> **Note**: These requirements align with [PICO's SecureMR documentation](https://developer-cn.picoxr.com/document/native/securemr-overview/). Ensure your environment matches these specifications before proceeding.

---

#### 2️⃣ Clone the Repository

```bash
git clone https://github.com/AarnavSan/SecureFabXR.git
cd SecureFabXR
```

---

#### 3️⃣ Open Project in Android Studio

1. Launch **Android Studio**
2. Select `File` → `Open`
3. Navigate to the **repository root directory** (the folder containing `settings.gradle`)
4. Click `OK`
5. Wait for **Gradle sync** to complete (this may take several minutes)
   - Gradle will download dependencies and configure the project
   - Check the Build window at the bottom for sync progress

> **Troubleshooting**: If Gradle sync fails, check that your NDK and SDK versions match the prerequisites.

---

#### 4️⃣ Prepare Your PICO 4 Ultra Device

1. **Enable Developer Mode**:
   - Put on your PICO 4 Ultra headset
   - Go to `Settings` → `General` → `About`
   - Tap `Software Version` **7 times** until developer mode is activated
   
2. **Enable USB Debugging**:
   - Go to `Settings` → `General` → `Developer`
   - Toggle on `USB Debugging`
   
3. **Connect Device to Computer**:
   - Connect headset to computer using USB-C cable
   - Put on the headset
   - You should see a prompt asking "Allow USB debugging?"
   - Check "Always allow from this computer"
   - Tap `OK`
   
4. **Verify Connection** (via terminal/command prompt):
   ```bash
   adb devices
   ```
   - You should see your device listed with status `device`
   - If device shows as `unauthorized`, check for the authorization prompt in headset

---

#### 5️⃣ Build and Deploy Application

**Option A: Deploy via Android Studio (Recommended)**

1. In Android Studio, locate the **run configuration dropdown** (near the top toolbar)
2. Select the `secure_fab` module from the dropdown
3. Ensure your PICO 4 Ultra is selected as the target device
4. Click the **Run button** (green play icon ▶️) or press `Shift + F10`
5. Wait for build to complete and app to install (first build may take 5-10 minutes)
6. App will automatically launch on your headset after installation

**Option B: Deploy via Command Line**

```bash
# Navigate to project root
cd SecureFabXR

# Build the debug APK
./gradlew :samples:secure_fab:assembleDebug

# Install to connected device
adb install -r samples/secure_fab/build/outputs/apk/debug/secure_fab-debug.apk

# Launch the application (optional)
adb shell am start -n com.pico.securefab/.MainActivity
```

> **Build Time**: First build typically takes 5-10 minutes as Gradle downloads dependencies and compiles native code. Subsequent builds are much faster.

---

#### 6️⃣ Grant Permissions and Launch

1. Put on your **PICO 4 Ultra** headset
2. Navigate to the app library
3. Find and launch **Secure Fab XR**
4. When prompted, grant **camera permissions** (required for object detection)
5. The application will initialize and begin detecting objects

---

### Verification

To verify successful installation:
- ✅ App launches without crashes
- ✅ Camera permission granted
- ✅ VST camera feed is active (you can see through the headset)
- ✅ Detection overlays appear when pointing at supported objects

If you encounter issues, see the [Troubleshooting](#-troubleshooting) section below.

---

## 🎮 Usage

### Operating the Application

1. **Launch the app** from your PICO 4 Ultra app library
2. **Grant camera permissions** when prompted
3. **Point your view at parts** on your workbench or in bins
4. **Observe detection overlays** appearing as 3D bounding boxes with labels
5. **Move your head naturally** - detections update in real-time as you move

### Demo Detection Classes

The default demonstration model detects common objects for testing purposes. **For actual industrial use, you should train and deploy custom models for your specific parts** (see Customization section).

Demo objects detected:
- Bottle
- Cup
- Scissors
- Book
- Keyboard
- Mouse
- Cell phone
- Monitor
- Laptop
- Pen

### Tips for Best Results

- **Good lighting** - Ensure adequate lighting in your work area
- **Clear view** - Position parts so they're not heavily occluded
- **Confidence threshold** - Adjust detection sensitivity based on your needs (default: 0.5)
- **Distance** - Parts should be within 0.5-3 meters for optimal detection
- **Part orientation** - Train models with parts in various orientations for robust detection

---

## 🔧 Customization

### Training and Deploying Custom Part Models

To detect your own industrial parts, you'll need to train a custom YOLOv11 model and convert it to SecureMR format.

#### Step 1: Prepare Training Data

1. Collect images of your industrial parts from various angles
2. Annotate images with bounding boxes and class labels
3. Use tools like [Roboflow](https://roboflow.com/) or [CVAT](https://www.cvat.ai/) for annotation
4. Export dataset in YOLO format

#### Step 2: Train YOLOv11 Model

```bash
# Using ultralytics library
pip install ultralytics

# Train the model
yolo detect train data=your_dataset.yaml model=yolov11n.pt epochs=100 imgsz=640

# Export to ONNX format
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

#### Step 3: Convert Model Using Docker

1. Navigate to the Docker directory:
   ```bash
   cd Docker
   ```

2. Place your ONNX model in the `models/` directory

3. Build and run the conversion container:
   ```bash
   docker build -t securemr-model-converter .
   docker run -v $(pwd)/models:/app/models securemr-model-converter
   ```

4. The converted model will be output as `model.serialized.bin`

#### Step 4: Deploy to Application

1. Replace the model file:
   ```bash
   cp Docker/models/model.serialized.bin samples/secure_fab/src/main/assets/yolom.serialized.bin
   ```

2. Update class labels in source code:
   - Edit `samples/secure_fab/cpp/yolo_object_detection.cpp`
   - Update the `CLASS_NAMES` array with your part names

3. Rebuild and deploy the application

### Adjusting Detection Parameters

Edit detection parameters in the configuration:

```cpp
// In yolo_object_detection.h or configuration file

// Confidence threshold (0.0 - 1.0)
const float DETECTION_THRESHOLD = 0.5;  // Increase for fewer false positives

// NMS IoU threshold (0.0 - 1.0)
const float NMS_THRESHOLD = 0.5;  // Decrease to keep more overlapping boxes

// Maximum detections to process
const int MAX_DETECTIONS = 20;  // Adjust based on your use case
```

---

## 🐛 Known Issues & Limitations

- **Lighting sensitivity** - Detection accuracy varies with lighting conditions
- **Occlusion handling** - Parts that are heavily occluded may not be detected
- **Multi-object performance** - Frame rate may drop with >10 simultaneous detections
- **Depth accuracy** - 3D positioning requires both stereo cameras to see the object clearly
- **Platform dependency** - Currently limited to PICO 4 Ultra

---

## 🔧 Troubleshooting

### Build Issues

**Problem**: Gradle sync fails with "NDK not found"
- **Solution**: Verify NDK 25.x.x is installed via SDK Manager

**Problem**: Build fails with "SDK Platform 34 not found"
- **Solution**: Install Android SDK Platform 34 via SDK Manager

### Deployment Issues

**Problem**: `adb devices` shows device as "unauthorized"
- **Solution**: Put on headset and check for USB debugging authorization prompt

**Problem**: App installs but crashes immediately on launch
- **Solution**: Check logcat output (`adb logcat`) for error messages

### Runtime Issues

**Problem**: No detections appearing
- **Solution**: Ensure adequate lighting and that objects are within 0.5-3 meters

**Problem**: Detection boxes appear in wrong locations
- **Solution**: Recalibrate headset tracking. Restart app if issue persists.

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 SecureFabXR Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Third-Party Components

This project incorporates several open-source libraries and components with their own licenses:

- **OpenXR SDK** - Licensed under Apache 2.0
- **PICO SecureMR SDK** - Licensed under ByteDance PICO Developer License
- **YOLOv11** - Licensed under AGPL-3.0 (commercial license available from Ultralytics)
- **Vulkan SDK** - Licensed under Apache 2.0

See [THIRD_PARTY_NOTICE](THIRD_PARTY_NOTICE) for complete third-party license information.

---

## 📖 Additional Resources

- [PICO SecureMR Documentation](https://developer-cn.picoxr.com/document/native/securemr-overview/)
- [SecureMR Resources](https://www.notion.so/SecureMR-Resources-2a982d330a5f806eb8adf01ad65c1868)
- [YOLOv11 Model Details](https://aihub.qualcomm.com/models/yolov11_det?searchTerm=yolo)
- [OpenXR Specification](https://www.khronos.org/openxr/)

---

## 🌟 Acknowledgments

- Developed as part of the **Stanford XR's Immerse the Bay** hackathon
- Built with **ByteDance PICO's SecureMR SDK**

---

**Built with ❤️ for the future of industrial XR**

*This is a demonstration project showcasing SecureMR capabilities for industrial applications. For production deployments, conduct thorough safety validation, testing, and compliance verification for your specific manufacturing environment.*
