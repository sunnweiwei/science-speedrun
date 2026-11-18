from pathlib import Path
import sys
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
from query_independence import compare_request_predictions


def test_row_order_and_extra_queries_do_not_change_profiles():
    x=sparse.csr_matrix([[1.,2.,3.],[3.,0.,4.]])
    result=compare_request_predictions({'a':x,'b':x},{'a':x[::-1]})
    assert result['changed_requests']==0


def test_query_conditioned_gene_shift_is_detected():
    x=sparse.csr_matrix([[1.,2.,3.],[3.,0.,4.]])
    y=x.toarray();y[:,0]*=1.3
    assert compare_request_predictions({'a':x},{'a':sparse.csr_matrix(y)})['changed_requests']==1


def test_changed_counts_or_unknown_request_are_not_compared():
    x=sparse.csr_matrix(np.ones((2,3)))
    with pytest.raises(ValueError):compare_request_predictions({'a':x},{'b':x})
    with pytest.raises(ValueError):compare_request_predictions({'a':x},{'a':x[:1]})
