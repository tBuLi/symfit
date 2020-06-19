import symfit as sf
import pytest

x, y, z = sf.variables('x, y, z')
a, b, c = sf.parameters('a, b, c')


@pytest.fixture(autouse=True)
def initVariales():
    global x, y, z
    x, y, z = sf.variables('x, y, z')
    

@pytest.fixture(autouse=True)
def initParameters():
    global a, b, c
    a, b, c = sf.parameters('a, b, c')