import torch
import pickle

class Embedding:
    def __init__(self, tensor: torch.Tensor, mask: torch.Tensor, scenario_ids: list[str]):
        self.rows = [
            {
                "scenario_id": scenario_id,
                "focal_embedding": tensor[i, 0, :].numpy(),
                "average_embedding": tensor[i, mask[i, :], :].numpy().mean(axis=0),
            } 
            for i, scenario_id in enumerate(scenario_ids)
        ]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.rows, f)
