import math

class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.deg = len(coeffs) - 1 # 1 + x + ... + x^n, len = n + 1
        self.msb = math.ceil(math.log(self.deg + 1, 2)) # min{j : degf < 2^j}

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            raise TypeError('Can only add other polynomials.')
        return [x + y for x in zip(self.coeffs, other.coeffs)]
        

    def __mul__(self, other):
        pass
