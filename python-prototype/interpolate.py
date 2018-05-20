

class Psi:
    """
    Operator for the FFT algorithm
    """
    def __init__(self, field, degree, shift):
        self.field = field
        self.shift = shift
        self.basis = self.find_basis(degree) # v_0, ..., v_{m-1}
        self.precomputed = [
            self.eval_s(j, shift) / self.eval_s(j, self.basis[j])
            for j in range(degree)
        ]

    def find_basis(self, degree):
        # 1, x, x^2, ... is a basis.
        # representation of elements is based on binary number corresponding to the polynomial.
        # so 1 = 001 = 1, 2 = 010 = x, 4 = 100 = x^2
        basis = [self.field(2**i) for i in range(4)] # 1, x, x^2, ... is a basis.
        
        # Check that it is a basis.
        r = range(2)
        coeffs = [[a, b, c, d] for a in r for b in r for c in r for d in r][1:]
        for vec in coeffs:
            result = sum([x*y for x,y in zip(basis, vec)])
            if(result == self.field(0)):
                raise ValueError("Not a basis: {}{}".format(result, vec))
        return basis

    def omega(self, i):
        # works because of how the basis is chosen, and because
        # of how the table is indexed.
        # e.g., elem 3 = 011 = 1 + x = b0 + b1
        return self.field(i)

    def eval_s(self, k, omega):
        """
        uses the recursive definition
        """
        if k == 0:
            return omega # s_0(x) = x
        # s_k(x) = s_{k-1}(x) * s_{k-1}(x - v_{k-1})
        return self.eval_s(k-1, omega) * self.eval_s(k-1, omega - self.basis[k-1])
        
    def apply(self, poly):
        """
        \psi_{\beta}(0,0) \leftarrow FFT_h(\Delta^0_0,\beta,0,0)
        h = deg poly
        """
        return self.apply_helper(poly.coeffs, poly.msb, self.shift)
        # return self.psi_helper(poly.coeffs, self.shift, 0, 0, poly.msb, basis.Delta(poly))

    def apply_helper(self, coeffs, k, beta):
        """
        Input: f(x) = (f_0, ..., f_{2^k - 1}), k = bin log of size, \beta = shift
        Output: 2^k evalusations, d_i = f(\omega + \beta)
        """
        if k == 0:
            return [coeffs[0]] # return f_0
        exp = 2**(k-1)
        recursive0 = []
        recursive1 = []
        for i in range(exp):
            g0 = coeffs[i] + self.precomputed[k-1] * coeffs[i + exp]
            g1 = g0 + coeffs[i + exp]
            recursive0.append(g0)
            recursive1.append(g1)

        result = self.apply_helper(recursive0, k-1, beta)
        result.extend(self.apply_helper(recursive1, k-1, beta))
        return result

        
    # def psi_helper(self, delta_r_i, beta, i, r, k, delta):
    #     """
    #     Input: FFT_h( \Delta^r_i ,\beta,i,r):  Delta^r_i is the recursion of the input polynomial, 
    #     h = 2^k denotes the size of the transform, and \beta\in F_{2^m}
    #     Output: \psi_{\beta}(i,r)={ r(\omega)|\omega\in V^k_i + \beta}
    #     V^k_j = span{v_j, \dots, v_{k-1}}
    #     """
    #     if i == k: # base case
    #         return delta_r_i # delta_r_i = {d_r}
    #     even = self.psi_helper(delta.coeffs(r, i+1), beta, i + 1, r, k - 1, delta) # 'even' parity subset
    #     odd = self.psi_helper(delta.coeffs(r+2**i, i+1), beta, i + 1, r + 2**i, k - 1, delta) # 'odd' parity subset
    #     resultset = []
    #     for j in range(2**(k - i - 1)):
    #         s = 1 # s poly
    #         result_even = even[j] + s*odd[j]
    #         result_odd  = result_even + odd[j]
    #         resultset.append(result_even) 
    #         resultset.append(result_odd)
    #     return resultset # check ordering
        

# class Delta:
#     """
#      Delta^r_i ={d_{j·2^i+r}|j =0,...,2^{k−i} −1)}.
#     """
#     def __init__(self, poly):
#         self.poly = poly

#     def coeffs(self, r, i):
#         """
#         coefficients of the midway point.
#         """
#         k = poly.msb
#         return [poly.coeffs[j*(2**i) + r] for j in range(2**(k - i)-1)]
