# Vehicle Counting and Traffic Analysis System

A robust vehicle detection, tracking, and traffic analysis system built using DETR (DEtection TRansformer) and DeepSORT algorithms. This system provides real-time vehicle counting, tracking, and traffic analysis capabilities.

## Features

- Real-time vehicle detection using DETR
- Multi-object tracking with DeepSORT
- Vehicle counting and classification
- Traffic analysis and statistics
- Visualization of detection and tracking results
- Support for video and image processing
- Traffic flow analysis and reporting

## Project Structure

```
DETR_vehicle_counting/
├── track1.py              # Main tracking and detection script
├── analyze_traffic.py     # Traffic analysis module
├── visualize_preds.py     # Visualization utilities
├── map_eval.py           # Mapping and evaluation tools
├── count_eval.py         # Counting evaluation module
├── train.py              # Training script for DETR model
├── csv_to_coco.py        # Data format conversion utility
├── deep_sort_pytorch/    # DeepSORT implementation
├── deepsort_config/      # DeepSORT configuration files
├── checkpoints/          # Model checkpoints
├── predictions/          # Output predictions
├── train/               # Training data
├── valid/               # Validation data
└── annotations/         # Dataset annotations
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- OpenCV >= 4.8.0
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd Vehicle_countingv2
```

2. Download DeepSORT:
   - Clone the DeepSORT repository:
   ```bash
   git clone https://github.com/ZQPei/deep_sort_pytorch.git
   ```
   - Copy the `deep_sort_pytorch` folder into the `DETR_vehicle_counting` directory
   - Download the DeepSORT weights from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
   - Place the downloaded weights in `DETR_vehicle_counting/deep_sort_pytorch/deep/checkpoint/`

3. Install dependencies:
```bash
pip install -r DETR_vehicle_counting/requirements.txt
```

## Usage

1. Run vehicle detection and tracking:
```bash
python DETR_vehicle_counting/track1.py --source [video_path]
```

2. Analyze traffic data:
```bash
python DETR_vehicle_counting/analyze_traffic.py --input [results_path]
```

3. Visualize predictions:
```bash
python DETR_vehicle_counting/visualize_preds.py --input [predictions_path]
```

## Model Training

To train the DETR model on your own dataset:

1. Prepare your dataset in COCO format
2. Configure training parameters in `train.py`
3. Run training:
```bash
python DETR_vehicle_counting/train.py
```

## Configuration

- DeepSORT parameters can be adjusted in the `deepsort_config` directory
- Model parameters can be modified in the respective script files
- Visualization settings can be customized in `visualize_preds.py`

## Output

The system generates:
- Real-time vehicle detection and tracking visualizations
- Traffic analysis reports
- Vehicle counting statistics
- JSON files containing detailed tracking data

## Acknowledgments

- DETR: [Facebook Research](https://github.com/facebookresearch/detr)
- DeepSORT: [Original Implementation](https://github.com/nwojke/deep_sort)
- DeepSORT PyTorch Implementation: [ZQPei's Implementation](https://github.com/ZQPei/deep_sort_pytorch) 