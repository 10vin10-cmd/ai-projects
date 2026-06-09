from ai_toolbox import add


def test_add_integers():
    assert add(1, 2) == 3


def test_add_floats():
    assert abs(add(1.5, 2.25) - 3.75) < 1e-9
