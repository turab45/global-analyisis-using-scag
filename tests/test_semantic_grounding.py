import numpy as np
import torch
from sklearn.metrics import average_precision_score

from run_semantic_grounding import spatial_ap, bootstrap_evidence


def test_ap_matches_sklearn_with_ties_constant_and_perfect_scores():
    scores = torch.tensor([[0., 0., 1., 1., 2., 2.], [1., 1., 1., 1., 1., 1.], [0., 1., 0., 1., 0., 1.]])
    labels = torch.tensor([[0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 0]])
    actual = spatial_ap(scores, labels).numpy()
    expected = [[average_precision_score(y, x) for y in labels.numpy()] for x in scores.numpy()]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert actual[2, 0] == 1
    assert torch.isnan(spatial_ap(scores, torch.zeros((1, 6)))).all()


def test_position_only_and_dead_channels_do_not_get_supported_edges():
    rng = np.random.default_rng(3)
    position = rng.uniform(.3, .5, (40, 1, 2))
    evidence = dict(ap=np.repeat(position, 3, axis=1), position=position, prevalence=np.full_like(position, .1), shifted=np.full((40, 3, 2), .1))
    mean, lower, consistency = bootstrap_evidence(evidence, 500, 42)
    assert not (lower > 0).all(axis=-1).any()
    evidence['ap'][:, 0] += .3
    mean, lower, consistency = bootstrap_evidence(evidence, 500, 42)
    assert (lower[0] > 0).all()
    assert not (lower[1:] > 0).all(axis=-1).any()
    assert (consistency[0] == 1).all()


def test_absent_concept_is_rejected_instead_of_getting_a_label():
    import pytest
    evidence = {key: np.full((3, 1, 1), np.nan) for key in ['ap', 'prevalence', 'position', 'shifted']}
    with pytest.raises(ValueError, match='no evaluable'):
        bootstrap_evidence(evidence, 100, 42)
