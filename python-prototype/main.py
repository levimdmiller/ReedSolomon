from interpolate import *
from model.poly import Polynomial
import numpy as np
import ff_factory

a = ff_factory.B4(15)
b = ff_factory.B4(12)
c = ff_factory.B4(3)
d = ff_factory.B4(6)

f = Polynomial([a, b, c, d])
x = Psi(ff_factory.B4, 4, ff_factory.B4(0))

print(x.apply(f))
    
