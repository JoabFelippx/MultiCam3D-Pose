import glob
import os

import cv2
import numpy as np

from skeletons import SkeletonsDetector


class VideoProcessor:
    
    def __init__(self, config: dict, flag="video", images_path=None, apply_undistort=True):
        self.config = config
        self.num_cameras = config["num_cameras"]
        self.flag = flag
        self.current_index = 0
        self.apply_undistort = apply_undistort 
        
        self.loop_images = True

        print(f"Carregando modelo YOLO: {config['yolo_model']}")
        detector_options = {"model": config["yolo_model"]}
        self.detector = SkeletonsDetector(detector_options)

        self.calib_data = [
            np.load(f"{config['calib_path']}/calib_rt{i}.npz")
            for i in range(1, self.num_cameras + 1)
        ]
        
        self.video_captures = []
        self.images = []

        if flag == "video":
            self.load_videos()
        if flag == "images":
            self.load_images(images_path)
         
    def load_videos(self):
        video_base_path = self.config["video_path"]
        for i in range(1, self.num_cameras + 1):
            video_path = os.path.join(video_base_path, f"camera_{i}_quatro.avi")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Não foi possível abrir o vídeo: {video_path}")
            self.video_captures.append(cap)

    def load_images(self, images_path):
        for camera_id in range(1, self.num_cameras + 1):
            imgs = []
            cam_idx = camera_id - 1
            patterns = [
                os.path.join(images_path, f"*cam{cam_idx}*"),
                os.path.join(images_path, f"*camera{cam_idx}*"),
                os.path.join(images_path, f"*_{cam_idx}_*"),
                os.path.join(images_path, f"*_{cam_idx}.*"),
            ]

            image_paths = []
            for p in patterns:
                image_paths = sorted(glob.glob(p))
                if image_paths:
                    break

            if not image_paths:
                search_path = os.path.join(images_path, "*")
                image_paths = sorted(glob.glob(search_path))

            if not image_paths:
                print(f"Aviso: Nenhum arquivo encontrado para {search_path}")

            for path in image_paths:
                print(path)
                img = cv2.imread(path)
                if img is not None:
                    imgs.append(img)
                else:
                    print(f"Aviso: Falha ao carregar a imagem {path}")

            self.images.append(imgs)

    def num_frames(self):
        if not self.images:
            return 0
        lengths = [len(imgs) for imgs in self.images]
        return min(lengths) if lengths else 0

    def _undistort_image(self, image: np.ndarray, cam_index: int) -> np.ndarray:
        npzCalib = self.calib_data[cam_index]
        
        K = npzCalib["K"]
        nK = npzCalib.get("nK", K)
        dist = npzCalib["dist"]
        
        undistort_img = cv2.undistort(image, K, dist, None, nK)
        
        if "roi" in npzCalib:
            x, y, w, h = npzCalib["roi"]
            if isinstance(npzCalib["roi"], np.ndarray):
                roi_arr = npzCalib["roi"]
                x, y, w, h = int(roi_arr[0]), int(roi_arr[1]), int(roi_arr[2]), int(roi_arr[3])
            return undistort_img[0:h, 0:w]
        
        return undistort_img

    def process_next_frame(self, count=-1):
        frames = []

        if self.flag == "video" and count == -1:
            all_frames_read = True
            for cap in self.video_captures:
                ret, frame = cap.read()
                if not ret:
                    all_frames_read = False
                    break
                frames.append(frame)

            if not all_frames_read:
                return None, None
        else:

            index = count if count != -1 else self.current_index

            for cam_idx in range(self.num_cameras):
                if len(self.images[cam_idx]) == 0:

                    return None, None

                if index < len(self.images[cam_idx]):
                    frame = self.images[cam_idx][index]
                else:
                    if self.loop_images:
                        frame = self.images[cam_idx][index % len(self.images[cam_idx])]
                    else:
                        return None, None

                frames.append(frame)
                
            if count == -1:
                self.current_index += 1

        if not frames:
            return None, None
        
        if self.apply_undistort:
            frames = [self._undistort_image(frame, i) for i, frame in enumerate(frames)]
        
        results_list = self.detector.detect(frames)

        annotations = []
        for i, results in enumerate(results_list):
            bboxes_keypoints_scores = results.keypoints.conf
            if bboxes_keypoints_scores is not None:
                bboxes_keypoints_scores = bboxes_keypoints_scores.cpu().numpy().astype("float32")
            else:
                obs_annotations = self.detector.to_object_annotations(
                    results, np.array([]), frames[i].shape
                )
                annotations.append(obs_annotations)
                continue

            obs_annotations = self.detector.to_object_annotations(
                results, bboxes_keypoints_scores, frames[i].shape
            )
            annotations.append(obs_annotations)
            
        return frames, annotations

    def release(self):
        for cap in self.video_captures:
            cap.release()