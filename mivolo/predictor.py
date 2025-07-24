from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
import huggingface_hub

class Predictor:
    def __init__(self, conf_threshold: float = 0.4, iou_threshold: float = 0.7, verbose: bool = False,
                  device: str = "cuda"):
        detector_weights = huggingface_hub.hf_hub_download('Ultralytics/YOLOv8', 'yolov8n.pt')
        #detector_weights = huggingface_hub.hf_hub_download('Ayah-kh/Mivolo-models', 'yolov8x_person_face.pt')
        checkpoint = huggingface_hub.hf_hub_download('Ayah-kh/Mivolo-models', 'mivolo_imbd.pth.tar')
        use_persons = True
        disable_faces = False
        draw = False
        self.detector = Detector(detector_weights, device, verbose=verbose,
                                 conf_thresh=conf_threshold, iou_thresh=iou_threshold)
        print(f"Detector initialized with weights: {detector_weights}")
        self.age_gender_model = MiVOLO(
            checkpoint,
            device,
            half=True,
            use_persons=use_persons,
            disable_faces=disable_faces,
            verbose=verbose,
        )
        self.draw = draw

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_objects, out_im

    def recognize_video(self, source: str, skip_frames: int = 0) -> Generator:
        """
        skip_frames : int, default 0
            Process only every ``skip_frames + 1``-th frame. 0 means process every frame.
        """
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)

        frame_idx = 0
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm.tqdm(range(total_frames)):
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                if not video_capture.grab():
                    break
                frame_idx += 1
                continue

            ret, frame = video_capture.read()
            if not ret:
                break

            detected_objects: PersonAndFaceResult = self.detector.track(frame)
            self.age_gender_model.predict(frame, detected_objects)

            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs

            # add tr_persons to history
            for guid, data in cur_persons.items():
                # not useful for tracking :)
                if None not in data:
                    detected_objects_history[guid].append(data)

            if self.draw:
                frame = detected_objects.plot()
            yield detected_objects_history, frame

    @staticmethod
    def summarize(
        detected_objects_history: Dict[int, List[Tuple[float, str]]]
    ) -> Dict[int, Dict[str, object]]:
        """
        Summarizes
        """
        id_stats = {}

        for track_id, records in detected_objects_history.items():
            ages = [age for age, _ in records]
            genders = [gender for _, gender in records]

            avg_age = sum(ages) / len(ages)
            male_count = sum(1 for g in genders if g.lower() == 'male')
            female_count = sum(1 for g in genders if g.lower() == 'female')
            total = len(genders)
            male_ratio = male_count / total if total > 0 else 0
            final_gender = 'male' if male_count > female_count else 'female'

            id_stats[track_id] = {
                'avg_age': round(avg_age, 2),
                'male_ratio': round(male_ratio, 2),  # 1.0 means 100% male, 0.5 = equal
                'final_gender': final_gender,
                'male_count': male_count,
                'female_count': female_count,
            }

        # Print nicely
        for tid, stats in id_stats.items():
            print(f"ID {tid}: Avg Age = {stats['avg_age']} | Gender = {stats['final_gender']} "
      f"({stats['male_count']}M/{stats['female_count']}F)")
