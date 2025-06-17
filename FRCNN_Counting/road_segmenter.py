import cv2


def get_vehicle_lane(vehicle_box, frame_width):
    x1, y1, x2, y2 = vehicle_box
    anchor_x = int((x1 + x2) / 2)

    if anchor_x < frame_width / 2:
        return "Left_Lane"
    else:
        return "Right_Lane"

def draw_lane_divider(frame):
    height, width = frame.shape[:2]
    color = (0, 255, 255)  # Yellow line
    thickness = 2
    return cv2.line(frame.copy(), (width // 2, 0), (width // 2, height), color, thickness)
