import numpy as np
import pytest
from run_argmax_labels import assign_labels


def test_forced_labels_preserve_weak_zero_and_tie_evidence():
    scores = np.array([[.01, .02, .03], [.001, .002, 0], [0, 0, 0], [.04, .04, 0]])
    result = assign_labels(scores, ['RV', 'MYO', 'LV'], .015)
    assert result.label.tolist() == ['LV', 'MYO', 'RV', 'RV']
    assert result.forced_below_threshold.tolist() == [False, True, True, False]
    assert result.tied_maximum.tolist() == [False, False, True, True]
    assert result.all_zero.tolist() == [False, False, True, False]
    np.testing.assert_allclose(result.margin, [.01, .001, 0, 0])


def test_invalid_scores_are_rejected():
    with pytest.raises(ValueError):
        assign_labels(np.array([[np.nan, 0]]), ['LV', 'RV'], .015)
