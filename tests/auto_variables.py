"""
This module is used to simplify tests by adding global variables and parameters. 
The idea is that you can use them everywhere without referencing:
In test cases, fixtures, or parametrize.
You can use the variables w, x, y, z, and the parameters a, b, c, d.
"""
import symfit as sf
import pytest

w, x, y, z = sf.variables('w, x, y, z')
a, b, c, d = sf.parameters('a, b, c, d')


@pytest.fixture(autouse=True)
def initVariales():
    """
    As autouse fixture this method re-initializes/resets the global variables w, x, y, z
    for every test, no matter if parametrized or not. 
    """
    global w, x, y, z
    w, x, y, z = sf.variables('w, x, y, z')


@pytest.fixture(autouse=True)
def initParameters():
    """
    As autouse fixture, this method re-initializes/resets the global parameters a, b, c, d
    for every test, no matter if parametrized or not. 
    """
    global a, b, c, d
    a, b, c, d = sf.parameters('a, b, c, d')
