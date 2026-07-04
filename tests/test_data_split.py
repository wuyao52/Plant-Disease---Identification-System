from __future__ import annotations

from plant_disease.data import split_indices_by_class


def test_split_indices_by_class_keeps_each_class_in_all_splits() -> None:
    targets = [0] * 10 + [1] * 10 + [2] * 10
    train, val, test = split_indices_by_class(targets, val_split=0.2, test_split=0.1, seed=7)
    assert len(train) == 21
    assert len(val) == 6
    assert len(test) == 3
    assert {targets[index] for index in train} == {0, 1, 2}
    assert {targets[index] for index in val} == {0, 1, 2}
    assert {targets[index] for index in test} == {0, 1, 2}
