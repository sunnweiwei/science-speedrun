import torch

from reference.alphagenome_50m.train_eval import PerTrackMoments


def test_pearson_retains_variance_for_large_offset_predictions():
    values = torch.tensor([[10_000.0], [10_001.0], [10_002.0]])
    moments = PerTrackMoments(1)
    moments.update(values, values)
    report = moments.report()
    assert report["valid_tracks"] == 1
    assert abs(report["mean_pearson"] - 1.0) < 1e-8
