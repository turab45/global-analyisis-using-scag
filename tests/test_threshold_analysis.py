import numpy as np
import pytest

from threshold_analysis import analyze_thresholds


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_threshold_sweep(capsys, dtype):
    # Two semantic pairs joined at 0.35, plus one neuron that is always isolated.
    R = np.array([[0, 0.8, 0, -0.5, 0], [0.8, 0, 0.35, 0, 0], [0, 0.35, 0, 0.8, 0], [-0.5, 0, 0.8, 0, 0], [0, 0, 0, 0, 0]], dtype=dtype)
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
    labeled = np.ones(5, dtype=bool)
    original = R.copy()
    results = analyze_thresholds(R, X, labeled, thresholds=[0, 0.35, 0.5, 0.8, 1])

    assert results["threshold"] == [0, 0.35, 0.5, 0.8, 1]
    assert results["n_edges"] == [3, 3, 2, 2, 0]
    assert results["n_nodes"] == [4, 4, 4, 4, 0]
    assert results["n_isolates"] == [1, 1, 1, 1, 5]
    assert results["n_components"] == [1, 1, 2, 2, 0]
    assert results["n_comm"] == [2, 2, 2, 2, 0]
    assert results["density"] == pytest.approx([0.3, 0.3, 0.2, 0.2, 0])
    assert results["largest_component_fraction"] == pytest.approx([0.8, 0.8, 0.4, 0.4, 0])
    assert results["mean_degree"] == [1.5, 1.5, 1, 1, 0]
    assert results["modularity"][2] == pytest.approx(0.5)
    assert results["ami"][2] == pytest.approx(1)
    assert np.isnan(results["modularity"][-1]) and np.isnan(results["ami"][-1])
    assert all(isinstance(values, list) and len(values) == 5 for values in results.values())
    assert "threshold = [0.0, 0.35, 0.5, 0.8, 1.0]" in capsys.readouterr().out
    np.testing.assert_array_equal(R, original)


def test_no_semantic_labels_and_default_threshold():
    results = analyze_thresholds(np.array([[0, 0.7], [0.7, 0]]), np.ones((2, 1)), np.zeros(2, dtype=bool))
    assert 0.35 in results["threshold"]
    assert all(np.isnan(value) for value in results["ami"])
    assert results["n_labeled"] == [0] * len(results["threshold"])


@pytest.mark.parametrize("thresholds", [[], [-0.1], [1.1], [np.nan], [np.inf]])
def test_invalid_thresholds(thresholds):
    with pytest.raises(ValueError, match="thresholds"):
        analyze_thresholds(np.eye(2), np.ones((2, 1)), np.ones(2, dtype=bool), thresholds=thresholds)


@pytest.mark.parametrize("R", [np.ones((2, 3)), np.array([[0, 1], [0, 0]]), np.array([[0, np.nan], [np.nan, 0]])])
def test_invalid_matrix(R):
    with pytest.raises(ValueError, match="R must"):
        analyze_thresholds(R, np.ones((2, 1)), np.ones(2, dtype=bool))
