import numpy as np
import pytest
import torch

from run_patient_semantic_labels import matched_evidence, residual_unit_maps


def test_fixed_location_alone_cannot_receive_a_label():
    template = torch.arange(30.).reshape(1, 30)
    maps = template[None].repeat(12, 1, 1)
    residual = residual_unit_maps(maps, template)
    assert not residual.any()
    result = matched_evidence(residual, residual, np.zeros(12), 199, 42)
    np.testing.assert_array_equal(result['effect'], 0)
    np.testing.assert_array_equal(result['p'], 1)


def test_detects_individual_spatial_variation_not_pathology_only():
    torch.manual_seed(19)
    values = torch.randn(24, 1, 40)
    x = residual_unit_maps(values, values.mean(0))
    result = matched_evidence(x, x, np.repeat([0, 1], 12), 499, 42)
    assert result['effect'].item() > .8
    assert result['p'].item() <= .01
    pathology = torch.cat([torch.zeros(12, 1, 40), torch.ones(12, 1, 40)], dim=0)
    # A group-specific spatial pattern still cannot beat within-group matching.
    pathology[:, :, ::2] *= -1
    y = residual_unit_maps(pathology, pathology.mean(0))
    result = matched_evidence(y, y, np.repeat([0, 1], 12), 199, 42)
    assert abs(result['effect'].item()) < 1e-6
    assert result['p'].item() == 1


def test_permutation_is_reproducible_and_singletons_are_rejected():
    torch.manual_seed(11)
    maps = torch.randn(8, 2, 20)
    x = residual_unit_maps(maps, maps.mean(0))
    first = matched_evidence(x, x, np.zeros(8), 199, 42)
    second = matched_evidence(x, x, np.zeros(8), 199, 42)
    np.testing.assert_array_equal(first['p'], second['p'])
    with pytest.raises(ValueError, match='at least two'):
        matched_evidence(x, x, np.arange(8), 199, 42)
