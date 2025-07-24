from collections import defaultdict
from typing import Dict, Generator, List

import numpy as np
from mivolo.model.mi_volo import MiVOLO
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
import huggingface_hub

class AgeGenderPredictor:
    def __init__(self, verbose: bool = False, device: str = "cuda"):
        checkpoint = huggingface_hub.hf_hub_download('Ayah-kh/Mivolo-models', 'mivolo_imbd.pth.tar')
        use_persons = True
        disable_faces = False
        self.age_gender_model = MiVOLO(
            checkpoint,
            device,
            half=True,
            use_persons=use_persons,
            disable_faces=disable_faces,
            verbose=verbose,
        )

    def predict_demographics(self, frame: np.ndarray, boxes: np.ndarray) -> Generator:

        detected_objects = PersonAndFaceResult(boxes)
        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)

        self.age_gender_model.predict(frame, detected_objects)

        current_frame_objs = detected_objects.get_results_for_tracking()
        cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs

        # add tr_persons to history
        for guid, data in cur_persons.items():
            if None not in data:
                detected_objects_history[guid].append(data)

        return detected_objects_history, frame
