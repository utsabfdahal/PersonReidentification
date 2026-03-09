# **Project: Person of Interest (POI) Tracker & Extractor**

## **🎯 Starting Prompt (Project Goal)**

"Develop an automated video processing pipeline to detect and track all unique individuals in a source video using YOLO-based detection and ByteTrack. The system must then perform Re-Identification (ReID) by comparing the 'visual fingerprint' of each tracked ID against a user-provided reference image of a Person of Interest (POI). Finally, the system should extract and concatenate all video segments where the POI is identified into a single output summary clip."

---

## **🏗 System Architecture**

The following diagram illustrates the flow from raw video input to the final POI-specific clip:

---

## **🛠 Prerequisites & Setup**

* **Language:** Python 3.9+  
* **Core Libraries:** `ultralytics` (YOLOv11), `opencv-python`, `torch`, `torchreid`, `moviepy`  
* **External Tool:** [FFmpeg](https://ffmpeg.org/) (Must be installed on your system path for video stitching)

### **Quick Install**

Bash  
pip install ultralytics opencv-python torchreid moviepy

---

## **📂 Project Structure**

Organize your workspace as follows to ensure the script paths function correctly:

* `input/`: Place your source video files here.  
* `ref/`: Place the `target.jpg` (POI reference image) here.  
* `output/`: This is where the final `.mp4` clips will be generated.  
* `main.py`: The primary execution script.

---

## **🚀 Execution Logic (The 4-Step Pipeline)**

### **Step 1: Detection & Tracking**

Utilize YOLOv11 for high-speed person detection. Use the `model.track()` method with **ByteTrack** enabled to maintain consistent IDs even when people temporarily overlap or cross paths.

### **Step 2: Feature Extraction (ReID)**

For every detected person-crop, use a pre-trained **OSNet** (Omni-Scale Network) to generate a feature embedding (a vector of 512 or 1024 numbers). Do the same for your `target.jpg`.

### **Step 3: Similarity Comparison**

Calculate the **Cosine Similarity** between the target's vector and the vectors of all tracked IDs.

* **Threshold:** If similarity \> 0.85, mark that Tracking ID as the POI.  
* **Smoothing:** Use a "Voting" system (e.g., if a person is matched in 10 consecutive frames, confirm them as the POI) to avoid flickering matches.

### **Step 4: Video Slicing**

Record the start and end timestamps where the POI is present. Use `moviepy` or `ffmpeg` to cut these segments and merge them.

---

## **🔧 Configuration Parameters**

Adjust these in your script to fine-tune performance:

* `SIMILARITY_THRESHOLD`: Default `0.80`. Increase for higher precision, decrease for higher recall.  
* `MIN_BOX_SIZE`: Ignore detections that are too small (far away) to be accurately identified.  
* `BUFFER_TIME`: Add 1–2 seconds of padding before and after a detection to make the final clip feel less "choppy."

