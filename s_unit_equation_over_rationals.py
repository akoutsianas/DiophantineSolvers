r"""
SageMath implementation of the resolution of the S-unit equation over the
rationals.

We solve the S-unit equation

        X + Y = 1,        X, Y  S-units of Q,

i.e. X, Y are non-zero rationals whose numerator and denominator have no
prime divisors outside the finite set S.  Equivalently, writing
X = x/z, Y = y/z with x + y = z and x, y, z pairwise coprime positive
S-integers, we solve the exponential equation

        x + y = z,        x, y, z  S-integers.           (0.1)

The resolution follows B. M. M. de Weger, "Solving exponential Diophantine
equations using lattice basis reduction algorithms", J. Number Theory 26
(1987), 325-367, and his PhD thesis "Algorithms for Diophantine Equations"
(Leiden, 1988), Chapter 6.  It consists of three steps:

1. An initial upper bound for the solutions is derived from lower bounds
   for linear forms in logarithms (Waldschmidt in the real case, van der
   Poorten in the p-adic case).  This is Theorem 5.1 of the paper, i.e.
   Theorem 6.1 of the thesis; it is implemented in
   ``SUnitEquationOverQ.initial_bound`` and gives an explicit integer
   bound for ``m(x*y*z) = max_i ord_{p_i}(x*y*z)``.

2. The bound is reduced with p-adic approximation lattices and the
   L^3-basis reduction algorithm (Section 5.B of the paper, Section 6.3
   of the thesis, Lemma 5.2).  This is implemented in
   ``SUnitEquationOverQ.reduce``: for each prime ``p`` the exponents of
   ``x*y*z`` at ``p`` are bounded by applying Lemma 5.2 to the lattice
   ``Gamma'_m``, and the procedure is iterated until the bound
   stabilises.

3. The remaining finitely many exponent vectors are enumerated.

Currently steps 1 and 2 are implemented.
"""

from sage.all import (ZZ, prod, log, exp, RealField, ceil, Qp, GF, matrix, gcd, lcm, xgcd, sqrt)
from copy import copy


class SUnitSolverOverQ:
    r"""
    Solve the S-unit equation  X + Y = 1  over Q.

    INPUT:

    - ``S`` -- a finite set (or iterable) of rational primes.
    - ``prec`` -- working precision (number of bits) used for the real
      computations (default: 200).
    - ``verbose`` -- if True, print progress information (default: False).

    EXAMPLES:

    Reproduce the bound of de Weger [1987], p. 350, for S = {2, 3, 5}::

        sage: import os, sys
        sage: sys.path.insert(0, os.getcwd())
        sage: from s_unit_equation_over_rationals import SUnitEquationOverQ
        sage: eq = SUnitEquationOverQ([2, 3, 5])
        sage: float(eq._constants(1, 3/8)["C10"]) < 6.76 * 10**41
        True
        sage: B = eq.initial_bound(mu=1, kappa=3/8)
        sage: B < 675 * 10**39
        True

    Reduce the initial bound (de Weger [1987], Section 5.B; the paper
    reports 56 for S = {2, 3, 5, 7, 11, 13})::

        sage: eq6 = SUnitEquationOverQ([2, 3, 5, 7, 11, 13])
        sage: eq6.reduce() < 100
        True
    """

    def __init__(self, S, prec=200, verbose=False):
        self.S = sorted(set(ZZ(p) for p in S))
        for p in self.S:
            if not p.is_prime():
                raise ValueError("S must consist of primes only")
        self.t = len(self.S)
        self.prec = prec
        self.verbose = verbose
        self.RR = RealField(prec)
        self._e = exp(self.RR(1))

        # Data of Theorem 5.1 that is independent of the parameters mu, kappa.
        self._s = (2 * self.t) // 3                    # s = [2t/3]
        self._P = prod(self.S)                         # P = p_1 * ... * p_t
        self._V = [max(self._e, log(self.RR(p)))       # V_i = max(e, log p_i)
                   for p in self.S]
        self._Omega = prod(self._V[self.t - self._s:])  # V_{t-s+1} ... V_t
        self._G = self._compute_G()

        self._bound_data = None

    def _compute_G(self):
        r"""
        G = max_i G_{p_i} / log p_i, where G_2 = 2, G_3 = 6 and
        G_p = p - 1 for p >= 5.
        """
        RR = self.RR

        def Gp(p):
            return RR(p * (p - 1)) if p in (2, 3) else RR(p - 1)

        return max(Gp(p) / log(RR(p)) for p in self.S)

    # ------------------------------------------------------------------ #
    # Theorem 5.1 (de Weger [1987]) / Theorem 6.1 (thesis)
    # ------------------------------------------------------------------ #
    def _constants(self, mu, kappa):
        r"""
        Return the constants of Theorem 5.1 for the chosen parameters.

        INPUT:

        - ``mu``, ``kappa`` -- real parameters with
          ``2/(s+1) <= mu <= 2`` and ``0 < kappa < mu/2``, where
          ``s = [2t/3]``.

        OUTPUT:

        A dictionary with the constants ``s, P, V, Omega, G, mu, kappa,
        eps, k, C6, C7, C8, C9, C10`` of the theorem, for which all
        primitive solutions of (0.1) satisfy ``m(x*y*z) < C10``.

        This is an internal helper; the public entry point is
        :meth:`initial_bound`.
        """
        RR = self.RR
        if self.t < 3:
            raise ValueError("Theorem 5.1 requires at least 3 primes")

        s = self._s
        mu = RR(mu)
        kappa = RR(kappa)
        if not (RR(2) / (s + 1) <= mu <= RR(2)):
            raise ValueError("mu must satisfy 2/(s+1) <= mu <= 2")
        if not (RR(0) < kappa < mu / 2):
            raise ValueError("kappa must satisfy 0 < kappa < mu/2")

        p1 = self.S[0]
        V = self._V
        Om = self._Omega
        G = self._G

        # C_6: Waldschmidt constant (Lemma 2.1 with n = s)
        C6 = RR(2) ** (9 * s + 26) * RR(s) ** (s + 4) \
            * Om * log(self._e * V[self.t - 2])

        # eps, k as in Theorem 5.1
        eps = (mu - kappa) / ((1 + kappa) * (1 + mu) * (s + 1))
        k = max(RR(16 * s) ** ((1 + 1 / kappa) * (s + 1)),
                (RR(8) / eps) ** ((1 + mu) * (s + 1)),
                RR(16) ** (1 / eps))

        # C_7: van der Poorten constant (Lemma 2.2 with n = s)
        C7 = RR(4 * (s + 1) ** (s + 1)) * k ** (1 + mu) * G * Om

        # C_8, C_9, C_10
        C8 = RR(4) * (C6 + C7 * log(RR(self._P) / RR(p1))) / log(RR(p1))
        C9 = C7 * log(C7) ** 2
        C10 = max(C9, C8 * log(C8) ** 2)

        return dict(s=s, P=self._P, V=V, Omega=Om, G=G, mu=mu, kappa=kappa,
                    eps=eps, k=k, C6=C6, C7=C7, C8=C8, C9=C9, C10=C10)

    def _minimize_c10(self):
        r"""
        Numerically minimize C_10 over the feasible parameters.

        The feasible set is ``2/(s+1) <= mu <= 2`` and
        ``0 < kappa < mu/2``.  This is only used to make the initial bound
        as small as possible; any feasible pair of parameters gives a
        rigorous bound.
        """
        RR = self.RR
        s = self._s
        mu_lo = RR(2) / (s + 1)
        mu_hi = RR(2)
        N = 25

        def scan(mu_lo, mu_hi):
            """Best (C10, mu, kappa) on an N x (N-1) grid."""
            best = None
            for i in range(N + 1):
                mu = mu_lo + (mu_hi - mu_lo) * RR(i) / N
                for j in range(1, N):
                    kappa = (mu / 2) * RR(j) / N
                    C10 = self._constants(mu, kappa)["C10"]
                    if best is None or C10 < best[0]:
                        best = (C10, mu, kappa)
            return best

        # coarse scan over the whole feasible set, then refine around mu
        _, mu, kappa = scan(mu_lo, mu_hi)
        for _ in range(3):
            step = (mu_hi - mu_lo) / N
            mu_lo = max(RR(2) / (s + 1), mu - step)
            mu_hi = min(RR(2), mu + step)
            _, mu, kappa = scan(mu_lo, mu_hi)
        return mu, kappa

    # ------------------------------------------------------------------ #
    # reduction (de Weger [1987], Section 5.B; thesis Section 6.3)
    # ------------------------------------------------------------------ #
    def _m0(self, p, q):
        r"""m_0 = ord_p(log_p(q)) for a prime q != p."""
        Kp = Qp(p, 100)
        return (Kp(q).log()).valuation()

    def _choose_p0(self, p, others):
        r"""
        Choose p_0 among ``others`` so that ord_p(log_p(p_0)) is minimal;
        among those, prefer a primitive root modulo p.

        The minimality of ord_p(log_p(p_0)) is what makes the numbers
        theta_i = -log_p(p_i)/log_p(p_0) p-adic integers.
        """
        Fp = GF(p)

        def key(q):
            return (self._m0(p, q),
                    0 if Fp(q).multiplicative_order() == p - 1 else 1,
                    q)

        return sorted(others, key=key)[0]

    def _theta(self, p, p0, others, m, m0):
        r"""
        Return theta_i^{(m)} = truncation of -log_p(p_i)/log_p(p_0)
        modulo p^m, for each p_i in ``others``.
        """
        prec = m + m0 + 20
        Kp = Qp(p, prec)
        lp0 = Kp(p0).log()
        out = []
        for q in others:
            theta = -(Kp(q).log() / lp0)
            out.append(theta.lift() % (p ** m))
        return out

    def _gamma_m(self, p, others, theta_m, m):
        r"""
        Return the basis matrix of the approximation lattice Gamma_m.

        The lattice is generated by the columns b_1,...,b_n,b_0 with
        b_i = e_i + theta_i^{(m)} e_{t-1} and b_0 = p^m e_{t-1}; a vector
        (x_1,...,x_n,x_0) lies in Gamma_m iff
        ord_p(x_1 theta_1 + ... + x_n theta_n - x_0) >= m.
        """
        n = len(others)
        B = matrix(ZZ, n + 1, n + 1)
        for i in range(n):
            B[i, i] = 1
            B[n, i] = theta_m[i]
        B[n, n] = p ** m
        return B

    def _gamma_prime_m(self, B, p, p0, others, theta_m, m):
        r"""
        Return the basis matrix of the sublattice Gamma'_m of Gamma_m
        whose vectors x satisfy p_0^{x_0}...p_n^{x_n} == +-1 mod p^{m+m0}.

        For p = 2, 3 the only roots of unity in Q_p are +-1, so
        Gamma'_m = Gamma_m.  For p > 3 a refinement is needed (the
        ``k(x)`` construction of the paper); it is simple when p_0 is a
        primitive root modulo p, and otherwise a Euclidean algorithm is
        applied to the discrete-log values k(b_i).
        """
        if p in (2, 3):
            return B
        n = len(others)
        Fp = GF(p)
        g = (p - 1) // 2

        if Fp(p0).multiplicative_order() == p - 1:
            # Primitive-root case: take the primitive (p-1)th root of
            # unity zeta = p0, so that k(b_0) = log_{p0}(p0) = 1 and the
            # Euclidean step is trivial.  Then k(b_i) = alpha_i +
            # theta_i^{(m)} (mod p-1) with p_i = p0^{alpha_i} (mod p),
            # and gamma_i = k(b_i) mod (p-1)/2.
            kvals = [Fp(pow(p0, theta_m[i], p) * (others[i] % p) % p)
                     .log(Fp(p0)) for i in range(n)]
            Bp = copy(B)
            for i in range(n):
                gamma = kvals[i] % g
                if gamma > g // 2:
                    gamma -= g
                Bp[:, i] = B[:, i] - gamma * B[:, n]
            Bp[:, n] = g * B[:, n]
            return Bp

        # General case (p0 not a primitive root mod p): k(x) mod (p-1)
        # is the discrete log in F_p^* of p_0^{x_0} p_1^{x_1} ... p_n^{x_n}
        # (mod p), taken with respect to a primitive root zeta; x lies in
        # Gamma'_m iff (p-1)/2 | k(x).  Apply the extended Euclidean
        # algorithm to the k-values k(b_i), moving the gcd to b_0 and
        # zeroing the rest, then take b*_i = b'_i and b*_0 = gamma_0 b'_0.
        zeta = Fp.primitive_element()
        kvals = [Fp(pow(p0, theta_m[i], p) * (others[i] % p) % p)
                 .log(zeta) for i in range(n)]
        kvals.append(Fp(p0).log(zeta))  # k(b_0)
        cols = [B[:, i] for i in range(n)] + [B[:, n]]
        kv = list(kvals)
        for i in range(n):
            a, b = kv[n], kv[i]
            gg = gcd(a, b)
            if gg == 0:
                kv[n], kv[i] = 0, 0
                continue
            _, x, y = xgcd(a, b)          # x*a + y*b = gg
            cn, ci = cols[n], cols[i]
            cols[n] = x * cn + y * ci     # k-value gg
            cols[i] = (-b // gg) * cn + (a // gg) * ci   # k-value 0
            kv[n], kv[i] = gg, 0
        k = kv[n]
        if k == 0:
            return B                      # Gamma'_m = Gamma_m
        gamma0 = lcm(k, g) // k
        Bp = matrix(ZZ, n + 1, n + 1)
        for i in range(n):
            Bp[:, i] = cols[i]
        Bp[:, n] = gamma0 * cols[n]
        return Bp

    def _lll_shortest_bound(self, B):
        r"""
        Lower bound for the length of the shortest nonzero vector of the
        lattice spanned by the columns of ``B`` (Lemma 3.1 of the paper).
        """
        RR = self.RR
        dim = B.ncols()
        c1 = B.transpose().LLL()[0]
        norm1 = RR(sqrt(sum(x * x for x in c1)))
        return norm1 / RR(2) ** ((dim - 1) / 2)

    def _reduce_one_prime(self, p, X0, max_m=10 ** 7):
        r"""
        Reduce the bound for ord_p(x*y*z).

        For the prime ``p`` and the current upper bound ``X0`` for
        m(x*y*z), build the lattice Gamma'_m and apply Lemma 5.2: if the
        shortest vector of Gamma'_m is longer than sqrt(t-1)*X0, then
        no solution has m + m0 <= ord_p(z) <= m(xyz) <= X0, hence
        ord_p(x*y*z) <= m + m0 - 1.  The smallest such ``m`` is found
        by binary search.
        """
        RR = self.RR
        t = self.t
        if t < 3:
            raise ValueError("reduction requires at least 3 primes")
        X0 = RR(X0)
        others = [q for q in self.S if q != p]
        p0 = self._choose_p0(p, others)
        m0 = self._m0(p, p0)
        others_i = [q for q in others if q != p0]
        n = t - 2

        def condition(m):
            theta_m = self._theta(p, p0, others_i, m, m0)
            B = self._gamma_m(p, others_i, theta_m, m)
            Bp = self._gamma_prime_m(B, p, p0, others_i, theta_m, m)
            bound = self._lll_shortest_bound(Bp)
            return bound ** 2 > (t - 1) * X0 ** 2

        # heuristic starting point, "somewhat larger" than the expected
        # size (t-1) log(sqrt(t-1) X0)/log p of m
        m_lo = 1
        m_hi = int(ceil((t - 1)
                        * (log(RR(sqrt(RR(t - 1)) * X0))
                           + (t - 2) / 2 * log(RR(2)))
                        / log(RR(p)))) + 5
        if m_hi < 1:
            m_hi = 1
        if m_hi > max_m:
            m_hi = max_m
        if not condition(m_hi):
            # not large enough: double until it works (or give up)
            while m_hi <= max_m and not condition(m_hi):
                m_hi = min(2 * m_hi, max_m)
        if not condition(m_hi):
            raise RuntimeError("reduction failed for p = %s" % p)
        # binary search for the smallest working m
        while m_hi - m_lo > 1:
            mid = (m_lo + m_hi) // 2
            if condition(mid):
                m_hi = mid
            else:
                m_lo = mid
        while not condition(m_hi):
            m_hi += 1
        return m_hi + m0 - 1

    def reduce_bound(self, X0):
        r"""
        One full reduction pass: return a new upper bound for
        m(x*y*z) obtained by applying Lemma 5.2 to every prime of S.
        """
        X0 = ZZ(X0)
        return max(self._reduce_one_prime(p, X0) for p in self.S)

    def reduce(self, X0=None):
        r"""
        Reduce the initial upper bound for m(x*y*z) until it stabilises.

        Starting from ``X0`` (default: the bound from
        :meth:`initial_bound`), repeatedly apply :meth:`reduce_bound`
        until the bound no longer decreases.
        """
        if X0 is None:
            X0 = self.initial_bound()
        X0 = ZZ(X0)
        while True:
            X1 = self.reduce_bound(X0)
            if X1 >= X0:
                break
            X0 = X1
        return X0

    def initial_bound(self, mu=None, kappa=None):
        r"""
        Return an integer ``B`` such that ``m(x*y*z) <= B`` for every
        primitive solution (x, y, z) of (0.1).

        This is Theorem 5.1 of de Weger [1987] (Theorem 6.1 of the
        thesis), giving ``m(x*y*z) < C_10``; the returned value is
        ``ceil(C_10) + 1`` (the ``+1`` absorbs the rounding error of the
        real computations).

        INPUT:

        - ``mu``, ``kappa`` -- parameters with ``2/(s+1) <= mu <= 2`` and
          ``0 < kappa < mu/2``.  If omitted, they are chosen numerically
          to minimize ``C_10`` (any feasible choice yields a rigorous
          bound).

        OUTPUT:

        An integer ``B`` with ``m(x*y*z) <= B`` for all primitive
        solutions, where ``m(a) = max_i ord_{p_i}(a)``.  In particular
        ``x*y*z <= P**B``, so ``log(x*y*z) <= B*log(P)``.
        """
        if mu is None or kappa is None:
            mu, kappa = self._minimize_c10()
        data = self._constants(mu, kappa)
        self._bound_data = data
        if self.verbose:
            print("C6 = %s" % data["C6"])
            print("C7 = %s" % data["C7"])
            print("C8 = %s" % data["C8"])
            print("C9 = %s" % data["C9"])
            print("C10 = %s" % data["C10"])
        return ZZ(ceil(data["C10"])) + 1
