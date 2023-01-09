# from src.tests import _PATH_DATA
import os
import os.path
import sys
import pytest
data_py_dir = os.path.join(os.path.dirname(__file__), '../data')
sys.path.append(os.path.abspath(data_py_dir))

data_dir = os.path.join(os.path.dirname(__file__), '../data2')
sys.path.append(os.path.abspath(data_dir))
print(data_dir)
# Import the data module
from data import mnist
bs = 63

@pytest.mark.skipif(not os.path.exists(data_dir), reason="Data files not found")
def test_data_length():
    dataset_test, _ = mnist()
    print(len(dataset_test)*bs)

    assert len(dataset_test)*bs > 25000 and len(dataset_test)*bs < 25064, "Dataset did not have the correct number of samples"
    # assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
    # assert that all labels are represented
