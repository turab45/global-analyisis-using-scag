from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scag.ablation import concept_ablation, predict_ablation
from scag.activations import extract_activations
from scag.matrices import compute_R
from scag.relations import adjacency, community_organisation, concept_rankings, concept_relations, concept_subgraph, fdr_bh, semantic_functional_pairs
from run_semantic_analysis import layer_names, sample_manifest


@pytest.fixture
def graph():
    R = np.array([[1, .9, .4, 0], [.9, 1, 0, 0], [.4, 0, 1, .8], [0, 0, .8, 1]])
    S = np.array([[1., 0], [1, 0], [0, 1], [0, 1]])
    return adjacency(R, S, .35), S


def test_projection_null_and_semantic_baseline(graph):
    A, S = graph
    table = concept_relations(A, S, ["LV", "MYO"], 99, 4)
    np.testing.assert_allclose(table.weight, [1.8, .4, 1.6])
    np.testing.assert_allclose(table.semantic_only, [2, 0, 2])
    assert (table.profile_p > 0).all() and (table.profile_q >= table.profile_p).all()
    assert (table.degree_q >= table.degree_p).all()
    assert table.equals(concept_relations(A, S, ["LV", "MYO"], 99, 4))
    np.testing.assert_allclose(table.profile_delta, table.weight - table.profile_null_mean)
    # Permuting complete profiles equals randomly relabelling graph nodes.
    perm = np.array([2, 0, 3, 1])
    inverse = np.argsort(perm)
    np.testing.assert_allclose(S[perm].T @ A @ S[perm], S.T @ A[np.ix_(inverse, inverse)] @ S)


def test_null_degenerate_and_empty_graph(graph):
    A, S = graph
    table = concept_relations(np.zeros_like(A), S, ["a", "b"], 9)
    assert (table.profile_p == 1).all() and table.profile_z.isna().all()
    enrichment, sve, partition = community_organisation(np.zeros_like(A), S, ["a", "b"], n_perm=9)
    assert enrichment.empty and partition.empty and np.isnan(sve.sve.iloc[0])


def test_queries_and_rankings(graph):
    A, S = graph
    nodes, edges = concept_subgraph(A, S, ["LV", "MYO"], "LV", neighbors=1)
    assert nodes.neuron.tolist() == [0, 1, 2]
    assert nodes.is_seed.tolist() == [True, True, False]
    assert list(zip(edges.source, edges.target)) == [(0, 1), (0, 2)]
    ranks, scores = concept_rankings(A, S, 0)
    np.testing.assert_allclose(scores["joint"], [0.9, .9, 0, 0])
    assert ranks["joint"][:2].tolist() == [0, 1]


def test_community_sve_and_enrichment(graph):
    A, S = graph
    enrichment, sve, partition = community_organisation(A, S, ["LV", "MYO"], n_perm=99)
    assert sve.sve.iloc[0] == pytest.approx(1)
    assert sve.null_mean.iloc[0] < 1
    assert len(enrichment) == 4 and len(partition) == 4
    assert enrichment.q.notna().all()


def test_pair_categories_exclude_zero_profiles(graph):
    A, S = graph
    pairs, summary = semantic_functional_pairs(A, S, n_perm=9)
    assert len(pairs) == 6
    assert summary["count"].tolist() == [3, 1, 0, 2]
    S[3] = 0
    pairs, _ = semantic_functional_pairs(A, S, n_perm=9)
    assert len(pairs) == 3 and 3 not in pairs.target.values


@pytest.mark.parametrize("bad", [np.array([[0, .2], [.1, 0]]), np.full((2, 2), np.nan), np.ones((2, 3))])
def test_invalid_correlations(bad):
    with pytest.raises(ValueError):
        adjacency(bad, np.ones((2, 1)), .35)


def test_bh_and_correlations_with_constant_channels():
    np.testing.assert_allclose(fdr_bh([.01, .04, .03]), [.03, .04, .04])
    acts = torch.tensor([[0., 1, 2], [0, 2, 1], [0, 3, 0]])[:, :, None, None]
    np.testing.assert_allclose(compute_R(acts), [[0, 0, 0], [0, 0, -1], [0, -1, 0]], atol=1e-12)
    np.testing.assert_allclose(compute_R(acts[:, 1:]), [[0, -1], [-1, 0]], atol=1e-12)
    assert compute_R(acts[:, :1]).shape == (1, 1)


def toy_model():
    model = nn.Sequential(nn.Identity(), nn.Flatten(), nn.Linear(2, 2, bias=False)).eval()
    with torch.no_grad():
        model[2].weight.copy_(torch.tensor([[1., 0], [0, 1]]))
    loader = DataLoader(TensorDataset(torch.tensor([[[[2.]], [[1.]]], [[[3.]], [[1.]]]])), batch_size=1)
    return model, loader


def test_ablation_equal_budgets_and_real_prediction_effect():
    model, loader = toy_model()
    A, S = np.array([[0., .9], [.9, 0]]), np.array([[1.], [.1]])
    table, patients = concept_ablation(model, model[0], loader, A, S, ["LV"], "cpu", (.5,), 2)
    assert set(table.strategy) == {"random", "functional", "semantic", "joint"}
    assert table.k.eq(1).all() and len(table) == 5 and len(patients) == 10
    assert table.loc[table.strategy == "semantic", "flip"].iloc[0] == 1
    assert table.kl.gt(0).all() and not model[0]._forward_hooks
    torch.testing.assert_close(predict_ablation(model, model[0], loader, 2, "cpu"), torch.softmax(torch.tensor([[2., 1], [3, 1]]), dim=1))


def test_hooks_cleanup_on_failure_and_snapshot_before_inplace():
    model, loader = toy_model()
    class Failing(nn.Module):
        def forward(self, x):
            raise RuntimeError("intentional")
    failing = nn.Sequential(nn.Identity(), Failing())
    with pytest.raises(RuntimeError, match="intentional"):
        predict_ablation(failing, failing[0], loader, 2, "cpu", [0])
    assert not failing[0]._forward_hooks
    model = nn.Sequential(nn.Identity(), nn.ReLU(inplace=True))
    loader = DataLoader(TensorDataset(-torch.ones(2, 1, 1, 1), torch.zeros(2, 2, 1, 1)))
    acts, _ = extract_activations(model, loader, model[0], "cpu")
    assert (acts == -1).all()


def test_layer_selection_and_patient_manifest():
    model, _ = toy_model()
    model.feature_layer_candidates = ("0", "2")
    assert layer_names(model, ["all"]) == ["0", "2"]
    assert layer_names(model, ["last"]) == ["2"]
    with pytest.raises(ValueError):
        layer_names(model, ["missing"])
    import pandas as pd
    dataset = SimpleNamespace(metadata=pd.DataFrame({"pid": ["p1", "p2"]}), data_dir="/data/train", split="train")
    manifest = sample_manifest(dataset, "train")
    assert manifest.pid.tolist() == ["p1", "p2"]
    assert manifest.sample_index.tolist() == [0, 1]


def test_plots_handle_empty_graph(tmp_path, graph):
    from scag.relation_plots import plot_layer
    A, S = graph
    A[:] = 0
    relations = concept_relations(A, S, ["LV", "MYO"], 9)
    pairs, _ = semantic_functional_pairs(A, S, n_perm=9)
    args = SimpleNamespace(tau=.35, gamma=.015, neighbors=3, similarity=.8, seed=42)
    plot_layer(relations, pairs, A, S, ["LV", "MYO"], args, tmp_path, "eval")
    assert (tmp_path / "relations_eval.png").is_file()
    assert (tmp_path / "evidence_subgraphs_eval.png").is_file()
