from ultralytics import YOLO
import cv2
import numpy as np

from is_msgs.image_pb2 import ObjectAnnotations

class SkeletonsDetector:

    def __init__(self, options):

        self.kpt_color_map = {
        0:{'name':'Nose', 'color':[0, 0, 255], 'radius':4},                
        1:{'name':'Right Eye', 'color':[255, 0, 0], 'radius':4},           
        2:{'name':'Left Eye', 'color':[255, 0, 0], 'radius':4},            
        3:{'name':'Right Ear', 'color':[0, 255, 0], 'radius':4},           
        4:{'name':'Left Ear', 'color':[0, 255, 0], 'radius':4},            
        5:{'name':'Right Shoulder', 'color':[193, 182, 255], 'radius':4},  
        6:{'name':'Left Shoulder', 'color':[193, 182, 255], 'radius':4},   
        7:{'name':'Right Elbow', 'color':[16, 144, 247], 'radius':4},      
        8:{'name':'Left Elbow', 'color':[16, 144, 247], 'radius':4},       
        9:{'name':'Right Wrist', 'color':[1, 240, 255], 'radius':4},       
        10:{'name':'Left Wrist', 'color':[1, 240, 255], 'radius':4},       
        11:{'name':'Right Hip', 'color':[140, 47, 240], 'radius':4},       
        12:{'name':'Left Hip', 'color':[140, 47, 240], 'radius':4},        
        13:{'name':'Right Knee', 'color':[223, 155, 60], 'radius':4},      
        14:{'name':'Left Knee', 'color':[223, 155, 60], 'radius':4},       
        15:{'name':'Right Ankle', 'color':[139, 0, 0], 'radius':4},        
        16:{'name':'Left Ankle', 'color':[139, 0, 0], 'radius':4},         
    }

        self.skeleton_map =  [
        {'srt_kpt_id':15, 'dst_kpt_id':13, 'color':[0, 100, 255], 'thickness':3},    
        {'srt_kpt_id':13, 'dst_kpt_id':11, 'color':[0, 255, 0], 'thickness':3},     
        {'srt_kpt_id':16, 'dst_kpt_id':14, 'color':[255, 0, 0], 'thickness':3},      
        {'srt_kpt_id':14, 'dst_kpt_id':12, 'color':[0, 0, 255], 'thickness':3},     
        {'srt_kpt_id':11, 'dst_kpt_id':12, 'color':[122, 160, 255], 'thickness':3},
        {'srt_kpt_id':5, 'dst_kpt_id':11, 'color':[139, 0, 139], 'thickness':3},    
        {'srt_kpt_id':6, 'dst_kpt_id':12, 'color':[237, 149, 100], 'thickness':3},  
        {'srt_kpt_id':5, 'dst_kpt_id':6, 'color':[152, 251, 152], 'thickness':3},    
        {'srt_kpt_id':5, 'dst_kpt_id':7, 'color':[148, 0, 69], 'thickness':3},        
        {'srt_kpt_id':6, 'dst_kpt_id':8, 'color':[0, 75, 255], 'thickness':3},        
        {'srt_kpt_id':7, 'dst_kpt_id':9, 'color':[56, 230, 25], 'thickness':3},       
        {'srt_kpt_id':8, 'dst_kpt_id':10, 'color':[0,240, 240], 'thickness':3},       
        {'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[224,255, 255], 'thickness':3},     
        {'srt_kpt_id':0, 'dst_kpt_id':1, 'color':[47,255, 173], 'thickness':3},    
        {'srt_kpt_id':0, 'dst_kpt_id':2, 'color':[203,192,255], 'thickness':3},    
        {'srt_kpt_id':1, 'dst_kpt_id':3, 'color':[196, 75, 255], 'thickness':3},     
        {'srt_kpt_id':2, 'dst_kpt_id':4, 'color':[86, 0, 25], 'thickness':3},        
        {'srt_kpt_id':3, 'dst_kpt_id':5, 'color':[255,255, 0], 'thickness':3},       
        {'srt_kpt_id':4, 'dst_kpt_id':6, 'color':[255, 18, 200], 'thickness':3}      
    ]
        self._model = None
        model_path = options.get('model', '')
    
        if model_path:
            self._model = YOLO(model_path)
            #self._model.to('cuda') # comentar para usar .engine
    def to_object_annotations(self, humans, kp_scores, image_shape):
        
        obs = ObjectAnnotations()
        bboxes_xyxy = humans.boxes.xyxy.cpu().numpy().astype('uint32')
        kp_scores_list = kp_scores.flatten().tolist()
        
        i = 0
        for bboxe_xyxy in bboxes_xyxy:
            obj = obs.objects.add()
            vertex_1 = obj.region.vertices.add()
            vertex_1.x = bboxe_xyxy[0]
            vertex_1.y = bboxe_xyxy[1]
            vertex_2 = obj.region.vertices.add()
            vertex_2.x = bboxe_xyxy[2]
            vertex_2.y = bboxe_xyxy[3]

            bbox_keypoints = humans.keypoints.data.cpu().numpy().astype('uint32')[i]
            for kpt_id in range(len(bbox_keypoints)):
                part = obj.keypoints.add()
                part.id = kpt_id + 1
                part.position.x = bbox_keypoints[kpt_id][0]
                part.position.y = bbox_keypoints[kpt_id][1]
                part.score = kp_scores_list[kpt_id]
            try:
                obj.id = int(humans[i].boxes.id)
                 
            except:
                continue
            obj.label = 'human'
            obj.score = humans.boxes.conf.cpu().numpy().astype('float32')[i]

            i+= 1
            
        obs.resolution.width = image_shape[1]
        obs.resolution.height = image_shape[0]

        return obs
        
    def to_object_annotation_3d(self, humans, image_shape=(1280, 720)):
        
        obs = ObjectAnnotations()
        
        for skeleton in humans:
            obj = obs.objects.add()

            for kpt_id, kpt in skeleton['skeleton_3d'].items():
                
                part = obj.keypoints.add()
                part.id = int(kpt_id)
                part.position.x = kpt[0]
                part.position.y = kpt[1]
                part.position.z = kpt[2]
                part.score = 1.0

            obj.label = 'human_3d'
            obj.score = 1.0
            obj.id = int(skeleton.get('id')) + 1 
            
        return obs
        
    def detect(self, images):
        results = self._model.track(images, device=0, verbose=False)
        return results