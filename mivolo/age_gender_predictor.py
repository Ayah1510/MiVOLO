from typing import Dict, Generator

import numpy as np
from mivolo.model.mi_volo import MiVOLO
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
import huggingface_hub
import torch

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

        detected_objects = PersonAndFaceResult([torch.tensor(box[:4]) for box in boxes])

        self.age_gender_model.predict(frame, detected_objects)

        current_frame_objs = detected_objects.get_results_for_tracking()
        age_gender_list = list(current_frame_objs.values())

        return age_gender_list
