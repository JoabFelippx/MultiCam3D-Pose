import os

import cv2
import numpy as np

from fundamental_matrices import FundamentalMatrices
from reconstructor_3d import Reconstructor3D
from skeleton_matcher import SkeletonMatcher

from utils import get_skeleton_center
from video_processor import VideoProcessor
from visualizer import Visualizer

# from streamChannel import StreamChannel
# from is_wire.core import Logger, Subscription, Message, Tracer

largura = 960
altura = 540
dim = (largura, altura)


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(script_dir, ".")

    config = {
        "calib_path": f"{base_dir}/calib_cameras",
        "video_path": f"{base_dir}/videos",
        "yolo_model": f"{base_dir}/models/yolo26m-pose.engine ",
        "num_cameras": 4,
        "n_keypoints": 18,
        
        #   Primeiro filtro
        "max_epipolar_dist": 15,  
        "min_matching_joints": 5,         
        #   Segundo filtro
        "max_intersection_dist": 7,  
        "weight_distance": 0.75,
        "weight_score": 0.25,
        #   Terceiro filtro
        "min_keypoints_for_grouping": 8,
    
        "kp_weights": {
            0: 1,  # Nose
            1: 0.2,  # Right Eye
            2: 0.2,  # Left Eye
            3: 0.2,  # Right Ear
            4: 0.2,  # Left Ear
            5: 1.0,  # Right Shoulder
            6: 1.0,  # Left Shoulder
            7: 1.0,  # Right Elbow
            8: 1.0,  # Left Elbow
            9: 1.0,  # Right Wrist
            10: 1.0,  # Left Wrist
            11: 1.0,  # Right Hip
            12: 1.0,  # Left Hip
            13: 1.0,  # Right Knee
            14: 1.0,  # Left Knee
            15: 1.0,  # Right Ankle
            16: 1.0,  # Left Ankle
        },
    }

    print("Inicializando módulos...")

    video_processor = VideoProcessor(config, images_path="OUTPUT_FRAMES")
    
    script_dir = os.path.dirname(__file__)
    calib_full_path = os.path.join(script_dir, config["calib_path"])
    camera_files = [
        f"{calib_full_path}/calib_rt{i}.npz"
        for i in range(1, config["num_cameras"] + 1)
    ]

    geometry = FundamentalMatrices(camera_files)
    projection_matrices = geometry.projection_matrices_all()
    fundamentals = geometry.fundamental_matrices_all()
    extrinsic_matrices = geometry.get_extrinsic_matrices()

    matcher = SkeletonMatcher(fundamentals, config)
    reconstructor = Reconstructor3D(projection_matrices, config)

    visualizer = Visualizer()

    print("Inicialização completa. Iniciando o loop principal...")

    while True:
        
        frames, annotations = video_processor.process_next_frame()
        
        if frames is None:
            break

        current_detections_3d = []
        skeletons_by_detection = []
        matched_2d_persons = []
        skeletons_to_visualize = []

        skeletons_2d, ids_2d = matcher.extract_skeletons_from_annotations(annotations)
        matched_persons = matcher.match(skeletons_2d, ids_2d, frames)
        
        reconstructed_skeletons = reconstructor.reconstruct_all(
            matched_persons, annotations
        )

        for idx, skeleton_data in enumerate(reconstructed_skeletons):
            if skeleton_data:
                hip_center = get_skeleton_center(skeleton_data)

                center_point_to_track = None

                if hip_center is not None:
                    center_point_to_track = hip_center
                else:
                    points_3d = np.array(list(skeleton_data.values()))
                    if len(points_3d) > 0:
                        center_point_to_track = np.mean(points_3d, axis=0)
                if center_point_to_track is None:
                    continue

                current_detections_3d.append(center_point_to_track)
                skeletons_by_detection.append(skeleton_data)
                person_match_ids = matched_persons[idx]['ids']
                matched_2d_persons.append(person_match_ids)

                skeletons_to_visualize.append(
                    {
                        "id": 1, # ainda não implementado :/ 
                        "skeleton_3d": skeleton_data,
                        "average_point": 0,
                        "matche_2d": 0,
                    }
                )


        plot_img_bgr = visualizer.update(frames, skeletons_to_visualize, extrinsic_matrices)
    
        h_small, w_small = 460, 300
        imgs_resized = [cv2.resize(f, (w_small, h_small), interpolation=cv2.INTER_NEAREST) for f in frames[:4]]
        
        linha1 = np.hstack(imgs_resized[:2])
        linha2 = np.hstack(imgs_resized[2:])
        grid = np.vstack((linha1, linha2))
        
        h_grid, w_grid = grid.shape[:2]
        h_plot, w_plot = plot_img_bgr.shape[:2]
        
        if h_plot != h_grid:
            scale = h_grid / h_plot
            new_w = int(w_plot * scale)
            plot_img_bgr = cv2.resize(plot_img_bgr, (new_w, h_grid), interpolation=cv2.INTER_NEAREST)

        combined_view = np.hstack((grid, plot_img_bgr))
        cv2.imshow("Combined View", combined_view)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
        
    video_processor.release()
    cv2.destroyAllWindows()
    print("Processamento finalizado.")


if __name__ == "__main__":
    main()
