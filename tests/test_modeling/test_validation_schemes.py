import numpy as np

from foodspec.modeling.validation.strategies import leave_one_group_out


def test_leave_one_group_out_no_leakage():
    groups = np.array(["A", "A", "B", "B", "C", "C"])
    for train_idx, test_idx in leave_one_group_out(groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)
