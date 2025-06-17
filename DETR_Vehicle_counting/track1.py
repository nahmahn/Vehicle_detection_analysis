import torch
import cv2
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection
from collections import defaultdict
import os
from tqdm import tqdm
import json

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


class VehicleTracker:
    def __init__(self, model_path="checkpoints/detr_epoch_10"):
        self.processor = DetrImageProcessor.from_pretrained(model_path)
        self.model = DetrForObjectDetection.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.deepsort = self.init_deepsort()

        self.track_history = defaultdict(lambda: [])
        self.max_history_length = 30  
        
        self.incoming_count = 0
        self.outgoing_count = 0
        self.counted_incoming_tracks = set()
        self.counted_outgoing_tracks = set()
        
        self.counting_line_y = 550  
        
        self.center_line_x = 500  
        
       
        self.colors = {
            1: (0, 255, 0),    
            2: (255, 0, 0),   
            3: (0, 0, 255),    
            4: (255, 255, 0),  
            5: (255, 0, 255)   
        }
        
        self.position_colors = {
            'left': (0, 255, 0),  
            'right': (255, 0, 0)   
        }
        
        self.deepsort_id_to_display_id = {}
        self.next_display_id = 0

        self.speed_events = [] 
        self.pixels_per_meter = 40.0  
        self.fps = 25  
        
        self.speed_flash_frames = {}  
        self.flash_duration = 10  

        self.analysis_data = []  

    def init_deepsort(self):
        cfg = get_config()
        config_path = "deepsort_config/deep_sort.yaml"
        reid_ckpt_path = "deepsort_config/ckpt.t7"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"DeepSORT config file not found: {config_path}")
        if not os.path.exists(reid_ckpt_path):
            raise FileNotFoundError(f"DeepSORT ReID checkpoint not found: {reid_ckpt_path}")

        cfg.merge_from_file(config_path)

        cfg.DEEPSORT.MIN_CONFIDENCE = 0.4
        cfg.DEEPSORT.NMS_MAX_OVERLAP = 0.7
        cfg.DEEPSORT.MAX_IOU_DISTANCE = 0.5
        cfg.DEEPSORT.MAX_DIST = 0.5
        cfg.DEEPSORT.MAX_AGE = 60
        cfg.DEEPSORT.N_INIT = 3
        cfg.DEEPSORT.NN_BUDGET = 1

        print("\n--- DeepSORT Params ---")
        print(f"ReID Checkpoint: {reid_ckpt_path}")
        print(f"Min Confidence: {cfg.DEEPSORT.MIN_CONFIDENCE}")
        print(f"Max IOU Distance: {cfg.DEEPSORT.MAX_IOU_DISTANCE}")
        print(f"Max Age: {cfg.DEEPSORT.MAX_AGE}")
        print(f"N Init: {cfg.DEEPSORT.N_INIT}")
        print(f"NN Budget: {cfg.DEEPSORT.NN_BUDGET}")
        print(f"Using CUDA: {self.device.type == 'cuda'}")
        print("------------------------\n")

        return DeepSort(
            reid_ckpt_path,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=self.device.type == 'cuda'
        )

    def process_frame(self, frame, frame_number):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb_frame, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.15
        )[0]

        detections = []
        detr_scores = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            detections.append([x1, y1, x2, y2, score.item(), label.item()])
            detr_scores.append(score.item())

        if detr_scores:
            print(f"DETR scores (min, max, avg): {min(detr_scores):.2f}, {max(detr_scores):.2f}, {sum(detr_scores)/len(detr_scores):.2f}")

        if not detections:
            return frame

        detections_np = np.array(detections)

        x_center = (detections_np[:, 0] + detections_np[:, 2]) / 2
        y_center = (detections_np[:, 1] + detections_np[:, 3]) / 2
        width = detections_np[:, 2] - detections_np[:, 0]
        height = detections_np[:, 3] - detections_np[:, 1]
        bbox_xywh = np.stack((x_center, y_center, width, height), axis=1).astype(np.float32)

        confs = detections_np[:, 4].astype(np.float32)
        clss = detections_np[:, 5].astype(np.float32)

        raw_outputs = self.deepsort.update(bbox_xywh, confs, clss, frame)

        outputs = raw_outputs[0] if isinstance(raw_outputs, tuple) else raw_outputs
        if not isinstance(outputs, np.ndarray) or outputs.size == 0:
            return frame

        valid_tracks = []
        for track_output in outputs:
            if isinstance(track_output, np.ndarray) and track_output.shape[0] == 6:
                valid_tracks.append(track_output)

        if not valid_tracks:
            return frame

        cv2.line(frame, (0, self.counting_line_y), (frame.shape[1], self.counting_line_y), (0, 0, 255), 2)
        cv2.putText(frame, "Speed Radar Line", (10, self.counting_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        current_frame_data = {
            'frame': frame_number,
            'incoming_count': self.incoming_count,
            'outgoing_count': self.outgoing_count,
            'tracks': []
        }

        for track_output in valid_tracks:
            x1, y1, x2, y2, actual_class_id, actual_track_id = track_output
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            actual_track_id, actual_class_id = int(actual_track_id), int(actual_class_id)
            
            if actual_track_id not in self.deepsort_id_to_display_id:
                self.deepsort_id_to_display_id[actual_track_id] = self.next_display_id
                self.next_display_id += 1
            
            display_track_id = self.deepsort_id_to_display_id[actual_track_id]

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            position = "left" if center[0] < self.center_line_x else "right"
            
            self.track_history[display_track_id].append(center)
            if len(self.track_history[display_track_id]) > self.max_history_length:
                self.track_history[display_track_id] = self.track_history[display_track_id][-self.max_history_length:]
            
            color = self.position_colors[position]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            cv2.putText(frame, f"ID: {display_track_id} ({position})", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if len(self.track_history[display_track_id]) >= 2:
                prev_y = self.track_history[display_track_id][-2][1]
                curr_y = center[1]
                
                if prev_y < self.counting_line_y and curr_y >= self.counting_line_y and display_track_id not in self.counted_incoming_tracks:
                    self.incoming_count += 1
                    self.counted_incoming_tracks.add(display_track_id)
                    
                    prev_center = self.track_history[display_track_id][-2]
                    curr_center = self.track_history[display_track_id][-1]
                    pixel_distance = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
                    
                    speed_kph = (pixel_distance * self.fps / self.pixels_per_meter) * 3.6
                    speed_kph = min(speed_kph, 250)  # Cap outliers
                    
                    self.speed_events.append({
                        'vehicle_id': display_track_id,
                        'frame': frame_number,
                        'speed_kph': round(speed_kph, 2),
                        'direction': 'incoming',
                        'position': position
                    })
                    
                    self.speed_flash_frames[display_track_id] = self.flash_duration
                
                elif prev_y > self.counting_line_y and curr_y <= self.counting_line_y and display_track_id not in self.counted_outgoing_tracks:
                    self.outgoing_count += 1
                    self.counted_outgoing_tracks.add(display_track_id)
                    
                    prev_center = self.track_history[display_track_id][-2]
                    curr_center = self.track_history[display_track_id][-1]
                    pixel_distance = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
                    
                    speed_kph = (pixel_distance * self.fps / self.pixels_per_meter) * 3.6
                    speed_kph = min(speed_kph, 250)  

                    self.speed_events.append({
                        'vehicle_id': display_track_id,
                        'frame': frame_number,
                        'speed_kph': round(speed_kph, 2),
                        'direction': 'outgoing',
                        'position': position
                    })
                    
                    self.speed_flash_frames[display_track_id] = self.flash_duration
            
            if display_track_id in self.speed_flash_frames:
                if self.speed_flash_frames[display_track_id] > 0:
                    cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 255), 3)
                    speed_text = f"Speed: {self.speed_events[-1]['speed_kph']:.0f} kph"
                    cv2.putText(frame, speed_text, (x1, y1 - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    self.speed_flash_frames[display_track_id] -= 1
                else:
                    del self.speed_flash_frames[display_track_id]
            
            if len(self.track_history[display_track_id]) > 1:
                points = np.array(self.track_history[display_track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, color, 2)
        
            current_frame_data['tracks'].append({
                'id': display_track_id,
                'bbox': [x1, y1, x2, y2],
                'position': position,
                'class_id': actual_class_id
            })

        self.analysis_data.append(current_frame_data)

        cv2.putText(frame, f"Incoming: {self.incoming_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Outgoing: {self.outgoing_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        current_deepsort_ids = set([t[5] for t in valid_tracks]) 

        deepsort_ids_to_remove_from_map = [k for k in self.deepsort_id_to_display_id if k not in current_deepsort_ids]
        for deepsort_id in deepsort_ids_to_remove_from_map:
            if deepsort_id in self.deepsort_id_to_display_id: 
                display_id_to_remove = self.deepsort_id_to_display_id[deepsort_id]
                if display_id_to_remove in self.counted_incoming_tracks:
                    self.counted_incoming_tracks.remove(display_id_to_remove)
                if display_id_to_remove in self.counted_outgoing_tracks:
                    self.counted_outgoing_tracks.remove(display_id_to_remove)
            del self.deepsort_id_to_display_id[deepsort_id]

        current_display_track_ids = set()
        for track_output in valid_tracks:
            actual_track_id = int(track_output[5])
            if actual_track_id in self.deepsort_id_to_display_id:
                current_display_track_ids.add(self.deepsort_id_to_display_id[actual_track_id])

        keys_to_delete_from_history = [key for key in self.track_history if key not in current_display_track_ids] 
        for key in keys_to_delete_from_history:
            del self.track_history[key]

        return frame


def main(video_path="C:/Users/namja/Downloads/traffic_test_data/traffic.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    os.makedirs("vis_preds", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("vis_preds/tracked_video2.mp4", fourcc, fps, (width, height))

    tracker = VehicleTracker()
    tracker.fps = fps 
    frame_number = 0 
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = tracker.process_frame(frame, frame_number) 
            out.write(processed_frame)
            pbar.update(1)
            frame_number += 1 

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Tracking complete. Output saved to vis_preds/tracked_video.mp4")

    analysis_output = {
        'total_incoming_vehicles': tracker.incoming_count,
        'total_outgoing_vehicles': tracker.outgoing_count,
        'speed_events': tracker.speed_events, 
        'frame_by_frame_data': tracker.analysis_data
    }
    with open("vis_preds/analysis_data.json", "w") as f:
        json.dump(analysis_output, f, indent=4)
    print("Analysis data saved to vis_preds/analysis_data.json")


if __name__ == "__main__":
    main()
