import pandas as pd
import pytest

from agbench.errors import ContractError
from agbench.manifests import check_input_window_isolation


def windows(rows):
    return pd.DataFrame(rows,columns=["chromosome","input_start","input_end"])


def test_full_flanks_overlap_although_centres_are_disjoint():
    train=windows([("chr1",0,1048576)])
    test=windows([("chr1",1000000,2048576)])
    with pytest.raises(ContractError,match="train/test: 1 overlapping"):
        check_input_window_isolation({"train":train,"test":test})


def test_touching_intervals_and_different_chromosomes_are_disjoint():
    splits={"train":windows([("chr1",0,1048576)]),
            "valid":windows([("chr2",0,1048576)]),
            "test":windows([("chr1",1048576,2097152)])}
    assert check_input_window_isolation(splits)=={"train/valid":0,"train/test":0,"valid/test":0}


@pytest.mark.parametrize("row", [(None, 0, 12), ("chr1", 12, 0), ("chr1", 0, 0),
                                  ("chr1", 0.5, 3.5)])
def test_malformed_windows_cannot_be_counted_as_disjoint(row):
    with pytest.raises(ContractError, match="invalid genomic input window"):
        check_input_window_isolation({"train": windows([row]), "test": windows([("chr1", 0, 12)])})
