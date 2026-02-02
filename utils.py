import cv2
import numpy as np

def get_skeleton_center(skeleton_3d: dict, hip_indices: list = [11, 12]):
    
    hip_points = [skeleton_3d.get(idx) for idx in hip_indices if skeleton_3d.get(idx) is not None]
    
    if len(hip_points) < 1:
        return None
    
    center_point = np.mean(np.array(hip_points), axis=0)
    return center_point
    
def draw_bounding_box(image, bbox_xyxy, color=(150, 0, 0), thickness=2):
    return cv2.rectangle(image, (int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3])), color, thickness)

def draw_identifier(image, bbox_xyxy, person_id, color=(150, 0, 0)):

    offset_x = 0
    offset_y = -10
    font_size = 1
    font_thickness = 2
    
    position = (int(bbox_xyxy[0] + offset_x), int(bbox_xyxy[1] + offset_y))
    
    return cv2.putText(image, str(person_id), position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)

def draw_skeleton(image, keypoints, skeleton_map):
    for connection in skeleton_map:
        srt_kpt_id = connection['srt_kpt_id']
        dst_kpt_id = connection['dst_kpt_id']
        
        p1 = keypoints[srt_kpt_id]
        p2 = keypoints[dst_kpt_id]

        if (p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0):
            continue

        color = connection.get('color', (0, 255, 0))
        thickness = connection.get('thickness', 2)

        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    return image

def draw_keypoints(image, keypoints, kpt_color_map):
    for kpt_id, data in kpt_color_map.items():
        point = keypoints[kpt_id]
        
        if point[0] == 0 and point[1] == 0:
            continue

        color = data.get('color', (0, 0, 255))
        radius = data.get('radius', 4)

        cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
    return image