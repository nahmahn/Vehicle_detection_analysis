import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image
import cv2
import numpy as np
import argparse
import json

#Utilities
import road_segmenter
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

def get_model_with_custom_anchors(num_classes):
    backbone = resnet_fpn_backbone('resnet50', weights='DEFAULT') #Resnet-50 with FPN


    #default aspect ratos are (0.5, 1.0, 2.0)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) #default
    aspect_ratios = ((0.4, 1.0, 2.5),) * len(anchor_sizes) # ratios favoring wider boxes (for testing video)

    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    #out_channels = backbone.out_channels

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator # Passing in our custom anchor generator
    )

    return model

def initialize_tracker():
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT, max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE, use_cuda=False)
    return deepsort


def process_video(video_path, model_path, output_path, line_y_fraction, ppm):
    device = torch.device('cpu')
    print(f"**Using device:{device}**")

    detection_model = get_model_with_custom_anchors(num_classes=2)
    detection_model.load_state_dict(torch.load(model_path, map_location=device))
    detection_model.to(device).eval()
    print("**Object detection model loaded successfully.**")

    tracker = initialize_tracker()
    print("**DeepSORT tracker initialized successfully.**")

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #All data structures to hold vehicle information for report generation.
    vehicle_data = {}
    track_history = {}
    speed_flash_info = {}
    speed_events = []
    max_congestion = {"Left_Lane": 0, "Right_Lane": 0}
    LANE_COLORS = {"Left_Lane": (255, 255, 0), "Right_Lane": (0, 165, 255)}
    counting_line_y = int(height * line_y_fraction)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (0, 255, 0), 2)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = detection_model(img_tensor)

        pred = predictions[0]
        conf_mask = pred['scores'] > 0.7
        all_boxes, all_scores, all_labels = pred['boxes'][conf_mask], pred['scores'][conf_mask], pred['labels'][conf_mask]

        outputs = []
        if len(all_boxes) > 0:
            xywh_boxes = [ [(box[0]+box[2])/2, (box[1]+box[3])/2, box[2]-box[0], box[3]-box[1]] for box in all_boxes.cpu().numpy() ]
            outputs, _ = tracker.update(np.array(xywh_boxes), all_scores.cpu().numpy(), all_labels.cpu().numpy(), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            tracker.increment_ages()
        current_frame_lane_counts = {"Left_Lane": 0, "Right_Lane": 0}

        if len(outputs) > 0: #TODO: Can use try except for better debugging.
            for output in outputs:
                x1, y1, x2, y2, track_cls, track_id = map(int, output)
                lane_name = road_segmenter.get_vehicle_lane((x1, y1, x2, y2), width)
                if lane_name in current_frame_lane_counts: current_frame_lane_counts[lane_name] += 1
                if track_id not in vehicle_data: #if id not available then add when it was first spotted and lane.
                    vehicle_data[track_id] = {"first_seen_sec": frame_count / fps, "lane_path": []}

                vehicle_data[track_id]["last_seen_sec"] = frame_count / fps
                if not vehicle_data[track_id]["lane_path"] or vehicle_data[track_id]["lane_path"][-1] != lane_name:
                    vehicle_data[track_id]["lane_path"].append(lane_name)

                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                if track_id not in track_history: track_history[track_id] = []
                track_history[track_id].append((center_x, center_y))
                if len(track_history[track_id]) >= 2:
                    prev_y = track_history[track_id][-2][1]
                    if (prev_y < counting_line_y and center_y >= counting_line_y) or (prev_y > counting_line_y and center_y <= counting_line_y):
                        #TODO: Use a set to check if the speed for this ID has already been logged.
                        if not any(event['track_id'] == track_id for event in speed_events):
                            pixel_distance = np.linalg.norm(np.array(track_history[track_id][-1]) - np.array(track_history[track_id][-2]))
                            speed_kph = (pixel_distance * fps / ppm) * 3.6

                            current_speed_event = {
                                "track_id": track_id,
                                "speed_kph": round(speed_kph, 2)
                            }
                            speed_events.append(current_speed_event)

                            speed_flash_info[track_id] = {"duration": 15, "speed": round(speed_kph, 2)}

                label = f"ID:{track_id} | {lane_name}"
                color = LANE_COLORS.get(lane_name, (255, 255, 255))
                # if track_id in speed_flash_info and speed_flash_info[track_id]["duration"] > 0:
                #     color = (0, 255, 255)
                #     speed_text = f"{speed_flash_info[track_id]['speed']:.1f} kph"
                #     cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                #     speed_flash_info[track_id]["duration"] -= 1

                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if track_id in speed_flash_info and speed_flash_info[track_id]["duration"] > 0:
                    color = (0, 255, 255)
                    speed_text = f"{speed_flash_info[track_id]['speed']:.1f} kph"
                    cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    speed_flash_info[track_id]["duration"] -= 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        max_congestion["Left_Lane"] = max(max_congestion["Left_Lane"], current_frame_lane_counts["Left_Lane"])
        max_congestion["Right_Lane"] = max(max_congestion["Right_Lane"], current_frame_lane_counts["Right_Lane"])

        writer.write(frame)
        frame_count += 1
        if frame_count % 30 == 0: print(f"Processed frame {frame_count}...")

    cap.release(), writer.release(), cv2.destroyAllWindows()

    json_output = generate_final_report(vehicle_data, speed_events, max_congestion)
    with open("traffic_summary.json", "w") as f:
        json.dump(json_output, f, indent=4)
    print("\n**JSON summary saved to traffic_summary.json**")
    print(json.dumps(json_output, indent=4))


def generate_final_report(vehicle_data, speed_events, max_congestion):
    speed_map = {event['track_id']: event['speed_kph'] for event in speed_events}

    vehicle_details = {}
    for track_id, data in vehicle_data.items():
        duration = round(data.get("last_seen_sec", 0) - data.get("first_seen_sec", 0), 2)
        final_speed = speed_map.get(track_id, 0)

        vehicle_details[str(track_id)] = {
            "duration_on_screen_sec": duration,
            "lane_path": data["lane_path"],
            "speed_at_tripwire_kph": final_speed
        }

    final_vehicle_details = {}
    last_known_ids = set(vehicle_data.keys())
    for track_id, details in vehicle_details.items():
        if int(track_id) in last_known_ids:
            final_vehicle_details[track_id] = details

    final_left_lane_count = sum(1 for data in final_vehicle_details.values() if "Left_Lane" in data["lane_path"])
    final_right_lane_count = sum(1 for data in final_vehicle_details.values() if "Right_Lane" in data["lane_path"])
    overall_summary = {
        "total_tracked_vehicles": len(final_vehicle_details),
        "unique_vehicles_in_left_lane": final_left_lane_count,
        "unique_vehicles_in_right_lane": final_right_lane_count,
        "max_left_lane_congestion": max_congestion["Left_Lane"],
        "max_right_lane_congestion": max_congestion["Right_Lane"]
    }
    final_json = {"overall_summary": overall_summary, "vehicle_data": final_vehicle_details}
    return final_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vehicle Tracking and Speed Analysis.")
    parser.add_argument('--video', required=True, help="Path to input video.")
    parser.add_argument('--model', required=True, help="Path to trained Faster R-CNN model (.pth).")
    parser.add_argument('--line-y', type=float, required=True, help="Y-coordinate of the speed measurement line as a fraction of frame height (e.g., 0.7 for 70% from the top).")
    parser.add_argument('--ppm', type=float, required=True, help="Pixels-per-meter ratio at the measurement line for accurate speed calculation.")
    parser.add_argument('--output', default="traffic_output.mp4", help="Path to save output video.")
    args = parser.parse_args()
    process_video(args.video, args.model, args.output, args.line_y, args.ppm)
