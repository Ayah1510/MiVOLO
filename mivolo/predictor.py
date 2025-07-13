from collections import defaultdict, Counter
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import math
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult



class Predictor:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose)
        print(f"Detector initialized with weights: {config.detector_weights}")
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

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
        face_to_person_map_history: Dict[int,  Optional[int]] = {}

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
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            # add tr_persons and tr_faces to history
            for guid, data in cur_persons.items():
                # not useful for tracking :)
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            updated_mapping = detected_objects.set_tracked_age_gender(detected_objects_history)
            face_to_person_map_history.update(updated_mapping )
            if self.draw:
                frame = detected_objects.plot()
            yield detected_objects_history, face_to_person_map_history, frame


    @staticmethod
    def summarize_person_detections(
        face_to_person_map_history_ids: Dict[int, Optional[int]],
        detected_objects_history: Dict[int, List[Tuple[float, str]]]
    ) -> Dict[int, Dict[str, object]]:
        """
        Summarizes age and gender predictions for each person based on associated face detections
        and their direct person ID entries in the detection history.

        Parameters:
        - face_to_person_map_history_ids: Maps face IDs to person IDs
        - detected_objects_history: Maps face or person IDs to lists of (age, gender) predictions

        Returns:
        - Dictionary mapping person ID to a summary with average age, majority gender, and face IDs
        """

        # Build mapping: person_id → face_ids
        person_to_face_ids = defaultdict(set)
        for face_id, person_id in face_to_person_map_history_ids.items():
            if person_id is not None:
                person_to_face_ids[person_id].add(face_id)

        # Gather detections only for persons who have face IDs
        person_to_ages_genders = defaultdict(list)

        for person_id, face_ids in person_to_face_ids.items():
            # 1. Get detections from face IDs
            for face_id in face_ids:
                for age, gender in detected_objects_history.get(face_id, []):
                    if not math.isnan(age):
                        person_to_ages_genders[person_id].append((age, gender))

            # 2. Include data from the person ID directly
            for age, gender in detected_objects_history.get(person_id, []):
                if not math.isnan(age):
                    person_to_ages_genders[person_id].append((age, gender))

        # Calculate age and gender per person
        summary = {}

        for person_id, entries in person_to_ages_genders.items():
            if not entries:
                continue

            ages = [age for age, _ in entries]
            genders = [gender for _, gender in entries]

            avg_age = round(np.mean(ages), 2) if ages else None
            majority_gender = Counter(genders).most_common(1)[0][0] if genders else None

            summary[person_id] = {
                "avg_age": avg_age,
                "majority_gender": majority_gender,
                "face_ids": list(person_to_face_ids[person_id])
            }

        return summary
