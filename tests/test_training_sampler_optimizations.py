"""Regression checks for the masked samplers used by periodic training evaluation."""

import torch
import torch.nn as nn

from scripts.masked_diffusion_finetune import maskgit_generate as continuation_sample
from scripts.mdm_sft_cfg2 import maskgit_generate as sft_sample


class _RecordingMaskedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 8)
        self.head = nn.Linear(8, 17)
        self.feature_batches = []
        self.projected_widths = []

    def predict_token_features(self, ids, _t):
        self.feature_batches.append(ids.shape[0])
        return self.embedding(ids)

    def project_token_features(self, features, positions=None):
        assert positions is not None
        self.projected_widths.append(positions.shape[1])
        selected = features.gather(
            1, positions[..., None].expand(-1, -1, features.shape[-1])
        )
        return self.head(selected)


def test_continuation_eval_projects_only_the_shrinking_unresolved_set():
    torch.manual_seed(0)
    model = _RecordingMaskedModel().eval()
    prompt = torch.randint(0, 16, (2, 3))

    output = continuation_sample(model, prompt, 5, 16, steps=4, top_k=3)

    assert output.shape == (2, 8)
    assert model.feature_batches == [2, 2, 2, 2]
    assert model.projected_widths == [5, 4, 3, 1]


def test_guided_sft_eval_batches_cfg_and_projects_only_unresolved_positions():
    torch.manual_seed(1)
    model = _RecordingMaskedModel().eval()
    prompt = torch.randint(0, 16, (2, 3))

    output = sft_sample(model, prompt, 5, 16, steps=4, top_k=3, guidance=2.5)

    assert output.shape == (2, 8)
    assert model.feature_batches == [4, 4, 4, 4]
    assert model.projected_widths == [5, 4, 3, 1]
