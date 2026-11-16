import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from scag.ablation import ablation_uncertainty, concept_ablation, matched_channels, predict_ablation, predict_interventions
from scag.relations import adjacency, community_organisation, concept_relations
from scag.research import bootstrap_relations, cache_patients, concept_evidence, validate_manifests
from scag.semantics import concept_masks, concept_profiles


def test_direct_mask_iou_and_frozen_thresholds():
    # ED and ES overlap at one pixel: union anatomy must count that pixel once.
    segments = torch.tensor([[[[1, 1], [0, 0]], [[1, 0], [0, 0]]]])
    masks, names = concept_masks(segments, 'anatomy')
    assert names == ['RV', 'MYO', 'LV'] and masks[0, 0].sum() == 2
    activations = torch.tensor([[[[1., 0.], [1., 0.]]]])
    result = concept_profiles(activations, masks, thresholds=np.array([.5]))
    assert result['intersection'][0, 0, 0] == 1
    assert result['union'][0, 0, 0] == 3
    assert result['S'][0, 0] == pytest.approx(1 / 3)
    phase_masks, _ = concept_masks(segments, 'anatomy_phase')
    phases = concept_profiles(activations, phase_masks, thresholds=result['thresholds'])
    assert phases['S'][0, 3] == pytest.approx(.5)
    np.testing.assert_array_equal(phases['thresholds'], [.5])
    # All-zero maps must never acquire anatomy labels.
    zero = concept_profiles(torch.zeros_like(activations), masks)
    assert not zero['S'].any()


def test_chunked_iou_aggregate_is_ratio_of_sums():
    acts = torch.tensor([[[[1., 0], [0, 0]]], [[[1., 1], [1, 1]]]])
    masks = torch.tensor([[[[1, 0], [0, 0]]], [[[1, 0], [0, 0]]]], dtype=torch.bool)
    result = concept_profiles(acts, masks, thresholds=np.array([.5]), batch_size=1, channel_batch=1)
    assert result['S'][0, 0] == pytest.approx(2 / 5)
    assert result['per_patient_iou'][:, 0, 0].tolist() == [1, .25]
    assert result['S'][0, 0] != result['per_patient_iou'].mean()
    manifest = pd.DataFrame({'pid': ['a', 'b'], 'sample_index': [0, 1]})
    evidence = concept_evidence(result, result['S'], ['RV'], manifest)
    assert evidence.iloc[0].pid == 'a' and evidence.iloc[0].patient_iou == 1


def test_semantics_do_not_change_functional_graph():
    R = np.array([[0, .4], [.4, 0]])
    np.testing.assert_array_equal(adjacency(R, np.zeros((2, 1)), .3), adjacency(R, np.ones((2, 3)), .3))


def test_rewire_null_and_bootstrap_are_seeded():
    A = np.array([[0, .8, 0, 0], [.8, 0, .4, 0], [0, .4, 0, .7], [0, 0, .7, 0]])
    S = np.array([[1., 0], [1, 0], [0, 1], [0, 1]])
    table = concept_relations(A, S, ['a', 'b'], 19, rewire=True)
    assert table.rewire_p.between(1 / 20, 1).all()
    assert (table.rewire_q >= table.rewire_p).all()
    assert table.equals(concept_relations(A, S, ['a', 'b'], 19, rewire=True))
    pooled = np.random.default_rng(1).normal(size=(10, 4))
    ci = bootstrap_relations(pooled, S, ['a', 'b'], .3, 19)
    assert (ci.weight_ci_low <= ci.weight_ci_high).all()
    assert ci.equals(bootstrap_relations(pooled, S, ['a', 'b'], .3, 19))


def test_fixed_training_partition_uses_new_semantics():
    A = np.array([[0, .9, 0, 0], [.9, 0, 0, 0], [0, 0, 0, .9], [0, 0, .9, 0]])
    S = np.array([[1., 0], [1, 0], [0, 1], [0, 1]])
    _, train, partition = community_organisation(A, S, ['a', 'b'], n_perm=9)
    _, heldout, same = community_organisation(A, S[[0, 2, 1, 3]], ['a', 'b'], n_perm=9, partition=partition)
    assert train.sve.iloc[0] == 1 and heldout.sve.iloc[0] == 0
    assert same.equals(partition)


class Patients(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, i):
        return torch.full((2, 2, 2), float(i)), torch.zeros(2, 2, 2), {'pid': f'p{i}', 'slice_index': i + 1}


def test_actual_patient_order_and_disjointness():
    manifest = pd.DataFrame({'pid': ['p0', 'p1', 'p2'], 'sample_index': range(3)})
    loader = cache_patients(Patients(), manifest, 2)
    assert len(loader.dataset) == 3 and manifest.slice_index.tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match='order'):
        cache_patients(Patients(), manifest.iloc[::-1], 2)
    with pytest.raises(ValueError, match='overlap'):
        validate_manifests({'train': manifest, 'test': manifest})
    with pytest.raises(ValueError, match='distinct'):
        validate_manifests({'train': pd.DataFrame({'pid': ['same', 'same']})})


def test_batched_interventions_equal_serial_and_cleanup():
    torch.manual_seed(9)
    model = nn.Sequential(nn.Conv2d(2, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(16, 3)).eval()
    loader = DataLoader(TensorDataset(torch.randn(5, 2, 2, 2)), batch_size=2)
    keys = [(), (0,), (1, 2), (0, 1, 2, 3)]
    batched = predict_interventions(model, model[0], loader, keys, 'cpu', 3)
    for key in keys:
        expected = predict_ablation(model, model[0], loader, 4, 'cpu', key)
        torch.testing.assert_close(batched[key], expected, rtol=1e-5, atol=1e-7)
    assert not model[0]._forward_hooks


def test_matched_controls_report_quality_without_replacement():
    covariates = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
    selected, gap, overlap = matched_channels(np.array([0, 1]), covariates, np.random.default_rng(42))
    assert set(selected) == {2, 3} and gap == 0 and overlap == 0
    _, gap, _ = matched_channels(np.array([0]), np.arange(12).reshape(4, 3), np.random.default_rng(42))
    assert gap > .1


def test_patient_uncertainty_does_not_treat_repeats_as_patients():
    rows = []
    for strategy, effect in [('joint', .3), ('semantic', .1)]:
        for i in range(10):
            rows.append({'concept': 'LV', 'budget': .1, 'k': 5, 'strategy': strategy, 'repeat': 0, 'sample_index': i, 'kl': effect + i / 100, 'flip': strategy == 'joint'})
    patients = pd.DataFrame(rows)
    summary, contrasts = ablation_uncertainty(patients, 499)
    duplicated = pd.concat([patients.assign(repeat=i) for i in range(5)])
    repeated, repeated_contrasts = ablation_uncertainty(duplicated, 499)
    pd.testing.assert_frame_equal(summary, repeated)
    pd.testing.assert_frame_equal(contrasts, repeated_contrasts)
    assert summary.n_patients.eq(10).all()
    kl = contrasts.query("metric == 'kl'").iloc[0]
    assert kl.joint_minus_control == pytest.approx(.2)
    assert kl.ci_low == pytest.approx(.2) and kl.p < .05


def test_all_selections_fixed_across_eval_values():
    model = nn.Sequential(nn.Identity(), nn.Flatten(), nn.Linear(4, 2)).eval()
    A = np.ones((4, 4)) - np.eye(4)
    S = np.array([[1.], [.9], [.2], [.1]])
    selections = []
    for value in [1., 2.]:
        loader = DataLoader(TensorDataset(torch.full((2, 4, 1, 1), value)), batch_size=1)
        table, _ = concept_ablation(model, model[0], loader, A, S, ['LV'], 'cpu', (.25,), 2, activation_magnitude=np.arange(4.))
        assert set(table.strategy) == {'joint', 'semantic', 'functional', 'random', 'matched', 'pagerank', 'activation'}
        assert table.k.eq(1).all()
        assert table.loc[table.strategy == 'matched', 'match_valid'].notna().all()
        selections.append(table[['strategy', 'repeat', 'neurons']])
    pd.testing.assert_frame_equal(*selections)
