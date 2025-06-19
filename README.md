# Vehicle Detection and Tracking System

A comprehensive vehicle detection, tracking, and traffic analysis system that implements two different approaches:
1. DETR (DEtection TRansformer) based implementation
2. Faster R-CNN based implementation
3. RF-DETR with ConvNext Backbone

First and Second implementations use DeepSORT for object tracking and provide similar functionality with different underlying detection architectures.

### RF-DETR

> ⚠️ **Note**: RF-DETR training was **not completed** due to hardware limitations. The model takes significantly longer to train compared to DETR and Faster R-CNN, especially when relying on Apple's MPS backend (MacBook M1/M2 GPU support).

When training on an M1 Mac, PyTorch often falls back to CPU for unsupported operations, leading to extremely slow performance. One such warning observed during training:
>>"UserWarning: The operator 'aten::grid_sampler_2d_backward' is not currently supported on the MPS backend
and will fall back to run on the CPU. This may have performance implications.
(Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/mps/MPSFallback.mm:14.)"


## Features

- Real-time vehicle detection using DETR, Faster R-CNN, or RF-DETR.
- Multi-object tracking with DeepSORT
- Vehicle counting and classification
- Traffic analysis and statistics
- Visualization of detection and tracking results
- Support for video and image processing
- Traffic flow analysis and reporting

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- OpenCV >= 4.8.0
- Other dependencies listed in respective `requirements.txt` files

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
   - Copy the `deep_sort_pytorch` folder into both `DETR_vehicle_counting` and `fasterRcnn_deepsort` directories
   - Download the DeepSORT weights from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
   - Place the downloaded weights in both:
     - `DETR_vehicle_counting/deep_sort_pytorch/deep/checkpoint/`
     - `fasterRcnn_deepsort/deep_sort_pytorch/deep/checkpoint/`

3. Download Model Weights:
   - DETR model weights: [Download](https://drive.google.com/file/d/1VQNx0iSiuvdir7M1TyhtGFw7q-dkXVeK/view?usp=drive_link)
   - Faster R-CNN model weights: [Download](https://drive.google.com/file/d/1amt5N8431WQErr-P-7d4XbGAAiKAnP_Y/view?usp=drive_link)
   - Place the DETR weights in `DETR_vehicle_counting/checkpoints/`
   - Place the Faster R-CNN weights in `fasterRcnn_deepsort/checkpoints/`

4. Install dependencies for DETR implementation:
```bash
pip install -r DETR_vehicle_counting/requirements.txt
```

5. Install dependencies for Faster R-CNN implementation:
```bash
pip install -r fasterRcnn_deepsort/requirements.txt
```

## Usage

### DETR Implementation

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

### Faster R-CNN Implementation

1. Run vehicle detection and tracking:
```bash
python fasterRcnn_deepsort/track.py --source [video_path]
```

## Model Training

### DETR Model
To train the DETR model on your own dataset:

1. Prepare your dataset in COCO format
2. Configure training parameters in `train.py`
3. Run training:
```bash
python DETR_vehicle_counting/train.py
```

### Faster R-CNN Model
The Faster R-CNN implementation uses a pre-trained model. If you want to fine-tune it:

1. Prepare your dataset in the required format
2. Configure training parameters in the respective script
3. Run training with your custom dataset

## Configuration

- DeepSORT parameters can be adjusted in the `deepsort_config` directory of each implementation
- Model parameters can be modified in the respective script files
- Visualization settings can be customized in the visualization scripts

## Output

Both implementations generate:
- Real-time vehicle detection and tracking visualizations
- Traffic analysis reports
- Vehicle counting statistics
- JSON files containing detailed tracking data

Sample outputs and results can be found in the [Outputs Folder](https://drive.google.com/drive/folders/1ALMeJpKJispp7a25wKNmcnxw-AS6m6L_?usp=drive_link)

## Choosing Between Implementations

- **DETR Implementation**: Better for modern architectures and transformer-based approaches
- **Faster R-CNN Implementation**: Better for traditional CNN-based approaches and potentially faster inference

## Acknowledgments

- DETR: [Facebook Research](https://github.com/facebookresearch/detr)
- DeepSORT: [Original Implementation](https://github.com/nwojke/deep_sort)
- DeepSORT PyTorch Implementation: [ZQPei's Implementation](https://github.com/ZQPei/deep_sort_pytorch)
- Faster R-CNN: [Original Paper](https://arxiv.org/abs/1506.01497) 
