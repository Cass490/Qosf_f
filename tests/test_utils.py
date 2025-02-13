#tests/test_utils.py

from utils import print_selected_readouts, get_predictor
import pytest
#from qrc_utils import my_utility_function  # Import your utility functions
import pytest

def test_print_selected_readouts(capfd): # capfd is a pytest fixture
    all_readouts = ["Z0", "Z1", "Z2", "Z3"]
    selected_indices = [0, 2]
    print_selected_readouts(selected_indices, all_readouts)
    out, err = capfd.readouterr()  # Capture the printed output
    assert out == "Selected Readouts:\n  Z0\n  Z2\n"
    assert err == ""  # Check that there's no error output

    # Test case with empty selection
    print_selected_readouts([], all_readouts)
    out, err = capfd.readouterr()
    assert out == "Selected Readouts:\n" # No readouts printed
    assert err == ""

def test_get_predictor():
    predictor = get_predictor()
    assert isinstance(predictor,RandomForestRegressor)