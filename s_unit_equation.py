r"""
SageMath solver for the S-unit equation over a number field.

We solve

        x + y = 1,        x in G_1,  y in G_2,

where `G_1` and `G_2` are finitely generated multiplicative subgroups of
`K^*`, `K` a number field.  The two groups need not be equal; each of them
is described by the generators of its free part (and, optionally, by
generators of its torsion part).

The resolution follows N. P. Smart, "The Algorithmic Resolution of
Diophantine Equations", LMS Student Texts 41, Cambridge University Press,
1998, Chapter IX (and the references therein), and is a class-based
refactoring of the reference implementation ``S-units over K.py``:

1. An initial upper bound for the exponents of a solution is obtained from
   lower bounds for linear forms in logarithms: Baker-Wustholz at the
   infinite places (:meth:`SUnitSolver.initial_bound_real_case`,
   :meth:`SUnitSolver.initial_bound_complex_case`) and Yu's theorem at the
   finite places (:meth:`SUnitSolver.initial_bound_finite_case`).

2. The bound is reduced at every place using LLL, following Smart's
   Lemmas VI.1, VI.2 and VI.5
   (:meth:`SUnitSolver.reduction_step_real_case`,
   :meth:`SUnitSolver.reduction_step_complex_case`,
   :meth:`SUnitSolver.reduction_step_finite_case`).

3. The remaining finite search is carried out by :meth:`SUnitSolver.solve`.

The low-level `p`-adic logarithm, embedding and closest-vector primitives
are those of :mod:`sage.rings.number_field.S_unit_solver`.

AUTHORS:

- Angelos Koutsianas
"""

from sage.all import (ZZ, QQ, RR, RealField, ComplexField, Integers, Infinity,
                      Integer, Set, log, exp, sqrt, pi, factorial, prod, mod,
                      vector, matrix, identity_matrix, zero_matrix, zero_vector,
                      block_matrix, polygen, ComplexField)
from sage.rings.number_field.S_unit_solver import (minimal_vector, log_p,
                                                   embedding_to_Kp)
from itertools import product as _iproduct
from copy import copy


class SUnitSolver:
    r"""
    Solve the `S`-unit equation `x + y = 1` over a number field `K`, with
    `x` in a finitely generated multiplicative group `G_1` and `y` in a
    finitely generated multiplicative group `G_2`.

    INPUT:

    - ``G1`` -- list of generators of the free part of `G_1` (elements of
      `K`).  A torsion generator may be included as well.
    - ``G2`` -- list of generators of the free part of `G_2`.
    - ``torsion1`` -- (optional) generators of the torsion part of `G_1`.
      If omitted, the roots of unity of `K` are used.
    - ``torsion2`` -- (optional) as ``torsion1`` for `G_2`.
    - ``prec`` -- working precision in bits (default: 200).
    - ``verbose`` -- if ``True``, print progress information.

    EXAMPLES::

        sage: import os, sys
        sage: sys.path.insert(0, os.getcwd())
        sage: from s_unit_equation import SUnitSolver
        sage: K.<a> = NumberField(x^2 + 5)
        sage: S = K.primes_above(30)
        sage: G = [g for g in K.S_unit_group(S=S).gens_values()
        ....:      if g.multiplicative_order() == Infinity]
        sage: sols = SUnitSolver(G, G).solve()
        sage: all(x + y == 1 for x, y in SUnitSolver(G, G).solve_pairs())
        True
    """

    def __init__(self, G1, G2, prec=200, verbose=False):
        if len(G1) == 0:
            raise ValueError('G1 is empty')
        if len(G2) == 0:
            raise ValueError('G2 is empty')

        if len([g for g in G1 + G2 if g == 0]) > 0:
            raise ValueError('The generators have to be non-zero')

        self.K = G1[0].parent()
        if not self.K.is_absolute():
            raise ValueError('K has to be an absolute number field')

        self.prec = prec
        self.verbose = verbose

        self._RR = RealField(prec)
        self._Kplaces = self.K.places(prec=self.prec)
        self._Krealplaces = self.K.real_embeddings(prec=self.prec)
        self._Kcomplexplaces = [s for s in self._Kplaces
                               if not self.is_real_place(s)]
        self._Kembeddings = self._Krealplaces + self._Kcomplexplaces

        self._G1free = [g for g in G1 if g.multiplicative_order() == Infinity]
        self._G2free = [g for g in G2 if g.multiplicative_order() == Infinity]

        self._torsion1 = self.K.roots_of_unity()
        self._torsion2 = self.K.roots_of_unity()

        self._g01 = self._torsion_generator(self._torsion1)
        self._g02 = self._torsion_generator(self._torsion2)

        self._G1all = ([self._g01] if self._g01 != 1 else []) + self._G1free
        self._G2all = ([self._g02] if self._g02 != 1 else []) + self._G2free

        self._initial_bound = None
        self._B = None
        self.supports = None
        self.c = None
        self._finite_init = None
        self._G2_ctx = None

        self._reduced_bound_G1 = None
        self._reduced_bound_G2 = None

    # ------------------------------------------------------------------ #
    # basic helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _round(a):
        r"""
        Return the closest integer to the real number ``a``.
        """
        if a in ZZ:
            return a
        return a.round()

    @staticmethod
    def _same_groups(G1, G2):
        r"""
        Return ``True`` if the two generator lists describe the same group.
        """
        if len(G1) != len(G2):
            return False
        return all(a == b for a, b in zip(G1, G2))

    def is_real_place(self, place):
        r"""
        Return ``True`` if the infinite ``place`` is real, otherwise
        ``False``.
        """
        prec = place.codomain().precision()
        return place.codomain() == RealField(prec)

    def higher_precision(self, place, new_prec):
        r"""
        Return the infinite ``place`` with precision raised to ``new_prec``
        bits.
        """
        old_prec = place.codomain().precision()
        Kplaces = self.K.places(prec=new_prec)
        gens = self.K.gens()

        i = 1
        while i <= old_prec:
            Q = [q for q in Kplaces
                 if len([0 for a in gens
                         if (q(a) - place(a)).abs() <= 2 ** (-i)]) == len(gens)]
            if len(Q) == 1:
                return Q[0]
            i += 1
        raise ValueError('I cannot find the place')

    def modified_height_infinite_case(self, a, place):
        r"""
        Return the modified absolute height of ``a`` at the infinite
        ``place``.

        REFERENCE:

        A. Baker and G. Wustholz, "Logarithmic forms and group varieties",
        J. Reine Angew. Math. 442 (1993), 19-62.
        """
        if a == 0 or a == 1:
            raise ValueError('a has not to be 0 or 1')

        precision = place.codomain().precision()
        K = place.domain()
        a = K(a)
        d = K.absolute_degree()
        height = max([d * (a.global_height(place.codomain().precision())),
                      log(place(a)).abs(), 1]) / d
        if RR(height) == Infinity:
            return self.modified_height_infinite_case(
                a, self.higher_precision(place, 2 * precision))
        return max([d * (a.global_height(place.codomain().precision())),
                    log(place(a)).abs(), 1]) / d

    def e_s_real(self, a, place):
        r"""
        Return ``a`` if ``place(a) >= 0`` and ``-a`` otherwise.
        """
        if place(a) < 0:
            return (-1) * a
        return a

    def Baker_Wustholz_low_lattice_bound(self, A, place):
        r"""
        Return the constant of the lower bound of Baker-Wustholz at
        ``place``.
        """
        if len(A) == 0:
            raise ValueError('The list A is empty')

        d = A[0].parent().absolute_degree()
        n = len(A)
        c = 18 * factorial(n + 1) * n ** (n + 1) \
            * (32 * d) ** (n + 2) * log(2 * n * d)
        return c * prod([self.modified_height_infinite_case(a, place)
                         for a in A])

    # ------------------------------------------------------------------ #
    # finite place helpers
    # ------------------------------------------------------------------ #
    def a_basis_with_0_order_at_p(self, prime, G):
        r"""
        Return the basis used in Lemma IX.3 of Smart's book.

        INPUT:

        - ``prime`` -- a prime ideal of a number field `K`
        - ``G`` -- a list of generators of a subgroup of `K^*`

        OUTPUT:

        a triple ``(M0, M, k)`` where ``M0`` contains all possible `\mu_0`,
        ``M`` contains all possible `\mu_i` (``i > 0``) and ``k`` is the
        index of the distinguished generator.
        """
        K = prime.ring()
        g0 = [g for g in G if g.multiplicative_order() != Infinity]
        Gfree = [g for g in G if g.multiplicative_order() == Infinity]
        if len(g0) == 1:
            g0 = g0[0]
        else:
            g0 = K(1)

        if len(Gfree) == 0:
            raise ValueError('The group does not have free part')
        if g0 == 0 or len([g for g in Gfree if g == 0]) != 0:
            raise ValueError('Either g0 = 0 or there is a zero element in G')

        e = prime.absolute_ramification_index()
        f = prime.residue_class_degree()
        ordprime = lambda x: x.valuation(prime)

        N = [ordprime(K(g)) for g in Gfree]
        n_k = min(N)

        if len([a for a in N if a != 0]) == 0:
            return [-g0 ** i for i in range(g0.multiplicative_order())], \
                Gfree, 0

        N_abs = [a.abs() for a in N]
        n_k_abs = min([a for a in N_abs if a > 0])
        k = N_abs.index(n_k_abs)
        n_k = N[k]
        N2 = [N[i] for i in range(len(N)) if i != k]
        G2 = [Gfree[i] for i in range(len(Gfree)) if i != k]

        M0 = []
        for vec in _iproduct(*([range(n_k_abs)] * len(N2))):
            sigma = -(vector(vec) * vector(N2))
            if sigma % n_k == 0:
                for g in [g0 ** i for i in range(g0.multiplicative_order())]:
                    m0 = -g * prod([a ** b for a, b in zip(G2, vec)]) \
                        * Gfree[k] ** (sigma / n_k)
                    if m0 not in M0:
                        M0.append(m0)
        return M0, [(g ** n_k) * (Gfree[k] ** (-n))
                    for g, n in zip(G2, N2)], k + 1

    def upper_bound_modified_height_finite_case(self, a, embeddings, prime):
        r"""
        Return an upper bound for the modified height with respect to
        ``prime``.

        REFERENCE:

        N. Tzanakis and B. M. M. de Weger, "Solving a specific Thue-Mahler
        equation", Math. Comp. 57 (1991), 799-815.
        """
        K = prime.ring()
        p = prime.absolute_norm().factor()[0][0]
        f = prime.residue_class_degree()
        d = K.absolute_degree()
        prec = embeddings[0].codomain().precision()

        t = polygen(K)
        if (p > 2 and not (t ** 2 + 1).is_irreducible()) or \
           (p == 2 and not (t ** 2 + 3).is_irreducible()):
            D = d
        else:
            D = 2 * d
        height = max([a.global_height(prec),
                      max([log(em(a)).abs() for em in embeddings])
                      / (2 * pi * D),
                      (f * log(p)) / d])
        if RR(height) == Infinity:
            return self.upper_bound_modified_height_finite_case(
                a, [self.higher_precision(em, 2 * prec) for em in embeddings],
                prime)
        return max([a.global_height(prec),
                    max([log(em(a)).abs() for em in embeddings])
                    / (2 * pi * D),
                    (f * log(p)) / d])

    def Yu_theorem(self, A, prime, embeddings):
        r"""
        Return the pair `(C_1C_2C_3, C_1C_2C_3C_4)` of Yu's theorem.

        REFERENCE:

        N. Tzanakis and B. M. M. de Weger, "Solving a specific Thue-Mahler
        equation", Math. Comp. 57 (1991), 799-815.
        """
        if len(A) == 0:
            raise ValueError('The list A is empty')

        if len([a for a in A if a.valuation(prime) != 0]) != 0:
            raise ValueError(
                'There is an element in A which does not have 0 valuation')

        K = prime.ring()
        d = K.absolute_degree()
        t = polygen(K)
        p = prime.absolute_norm().factor()[0][0]
        n = len(A)
        f = prime.residue_class_degree()

        D = 2 * d
        if (p > 2 and not (t ** 2 + 1).is_irreducible()) or \
           (p == 2 and not (t ** 2 + 3).is_irreducible()):
            D = d

        if p % 4 == 1:
            c1 = 35009 * (45 / 2) ** n
        elif p % 4 == 3:
            c1 = 30760 * 25 ** n
        else:
            c1 = 197142 * 36 ** n

        c4 = 2 * log(D)
        v = [self.upper_bound_modified_height_finite_case(a, embeddings, prime)
             for a in A]
        V = max(v)

        if p != 2:
            c3 = log(2 ** 11 * (n + 1) ** 2 * D ** 2 * V)
        else:
            c3 = log(3 * 2 ** 10 * (n + 1) ** 2 * D ** 2 * V)
        c2 = (n + 1) ** (2 * n + 4) * p ** ((D * f) / d) \
            * (f * log(p)) ** (-n - 1) * D ** (n + 2) * prod(v)

        return c1 * c2 * c3, c1 * c2 * c3 * c4

    # ------------------------------------------------------------------ #
    # support and membership
    # ------------------------------------------------------------------ #
    def support_of_G(self, G):
        r"""
        Return the support of the group generated by ``G``.

        OUTPUT:

        a triple ``(finite, real, complex)`` of finite primes, real places
        and complex places occurring in the support.
        """
        if len(G) == 0:
            raise ValueError('G is empty')

        complexsup = [c for c in self._Kcomplexplaces
                      if len([a for a in G
                              if (c(a).abs() - 1).abs()
                              >= 2 ** (-self.prec / 2)]) > 0]
        realsup = [c for c in self._Krealplaces
                   if len([a for a in G
                           if (c(a).abs() - 1).abs()
                           >= 2 ** (-self.prec / 2)]) > 0]

        finitesup = []
        for g in G:
            for p in g.support():
                if p not in finitesup:
                    finitesup.append(p)

        rational_primes_below = [p.absolute_norm().factor()[0][0]
                                 for p in finitesup]
        for i in range(len(finitesup)):
            for j in range(i, len(finitesup)):
                if rational_primes_below[j - 1] > rational_primes_below[j]:
                    rational_primes_below[j - 1], rational_primes_below[j] = \
                        rational_primes_below[j], rational_primes_below[j - 1]
                    finitesup[j - 1], finitesup[j] = \
                        finitesup[j], finitesup[j - 1]
        return finitesup, realsup, complexsup

    def is_S_unit_element(self, SUK, u):
        r"""
        Return ``True`` if ``u`` is an ``S``-unit for the group ``SUK``.

        INPUT:

        - ``SUK`` -- an ``S``-unit group of a number field `K`
        - ``u`` -- an element of `K`
        """
        K = SUK.number_field()
        try:
            u = K(u)
        except (TypeError, ValueError):
            raise ValueError("%s is not an element of %s" % (u, K))

        if u == 0:
            return False
        return K.ideal(u).is_S_unit(list(SUK.primes()))

    def _group_context(self, G):
        r"""
        Build the data needed to test membership in the group generated by
        ``G``, and return ``(SunitK, A, supp)``.
        """
        K = G[0].parent()
        supp = self.support_of_G(G)[0]
        SunitK = K.S_unit_group(S=supp)
        m0 = min([g.multiplicative_order() for g in SunitK.gens()])
        m = len(SunitK.group_generators())
        k = len(G)

        A = copy(zero_matrix(ZZ, k + 1, m))
        for i, g in enumerate(G):
            A[i] = SunitK(g).list()
        A[k] = zero_vector(ZZ, m)
        A[k, 0] = m0
        return SunitK, A, supp

    def is_in_G(self, x, G):
        r"""
        Return ``True`` if the non-zero element ``x`` lies in the group
        generated by ``G``.
        """
        if x == 0:
            raise ValueError('x is the zero element')
        if len(G) == 0:
            raise ValueError('G is empty')

        K = G[0].parent()
        SunitK, A, supp = self._group_context(G)

        if K.ideal(x).is_S_unit(S=supp):
            y = vector(SunitK(x).list())
            return y in A.row_space()
        return False

    # ------------------------------------------------------------------ #
    # constants c_1, c_2, c_3
    # ------------------------------------------------------------------ #
    def c_constants(self, G, precision):
        r"""
        Return the constants `c_1, c_2, c_3` of page 136 of Smart's book.

        The computation is checked against a doubling of the precision.
        """
        Real = RealField(prec=precision)
        c1, c2, c3 = self.c_constants_without_check(G, precision)
        c1d, c2d, c3d = self.c_constants_without_check(G, 2 * precision)
        if (c1 - c1d).abs() > 1:
            return Real(c1d), Real(c2d), Real(c3d)
        return c1, c2, c3

    def c_constants_without_check(self, G, precision):
        r"""
        Return the constants `c_1, c_2, c_3` of page 136 of Smart's book.
        """
        if len(G) == 0:
            raise ValueError('G is empty')
        if len([g for g in G if g.multiplicative_order() != Infinity]) > 0:
            raise ValueError('G has an element with finite multiplicative order')

        finiteSup, realSup, complexSup = self.support_of_G(G)

        if len([s for s in realSup if not self.is_real_place(s)]) > 0:
            raise ValueError('realSup has a complex place')
        if len([s for s in complexSup if self.is_real_place(s)]) > 0:
            raise ValueError('complexSup has a real place')

        if len(realSup) > 0:
            K = realSup[0].domain()
        elif len(complexSup) > 0:
            K = complexSup[0].domain()
        else:
            K = finiteSup[0].ring()

        A = copy(zero_matrix(RealField(precision),
                             len(finiteSup) + len(complexSup) + len(realSup),
                             len(G)))

        for i, p in enumerate(finiteSup):
            v = [0] * len(G)
            for j, g in enumerate(G):
                if (K(g)).abs_non_arch(p, prec=precision) != 1:
                    v[j] = log((K(g)).abs_non_arch(p, prec=precision))
            A[i] = vector(v)

        for i, s in enumerate(realSup):
            v = [0] * len(G)
            for j, g in enumerate(G):
                if abs(s(g)) != 1:
                    v[j] = log(abs(s(g)))
            A[i + len(finiteSup)] = vector(v)

        for i, s in enumerate(complexSup):
            v = [0] * len(G)
            for j, g in enumerate(G):
                if abs(s(g)) != 1:
                    v[j] = 2 * log(abs(s(g)))
            A[i + len(finiteSup) + len(realSup)] = vector(v)

        n = len(finiteSup) + len(complexSup) + len(realSup)
        s = Set(range(n))
        X = s.subsets(len(G)).list()
        c1 = -Infinity
        for g in X:
            M = A[g.list(), :]
            d = M.determinant()
            if d > 2 ** (-RR(precision / 2).floor()):
                B = M.inverse()
                a = max([sum([b.abs() for b in row]) for row in B.rows()])
                if a > c1:
                    c1 = a

        c2 = 1 / c1
        c3 = c2 / len(G)
        c3 = (99 * c3) / 100

        return c1, c2, c3

    # ------------------------------------------------------------------ #
    # initial bounds (step 1)
    # ------------------------------------------------------------------ #
    def initial_bound_real_case(self, G2free, place, c3):
        r"""
        Return the initial bound of Lemma IX.1.1 at a real ``place``.
        """
        if not self.is_real_place(place):
            raise ValueError('place is not real')

        c5 = log(2) / c3
        c7 = c3
        c8 = self.Baker_Wustholz_low_lattice_bound(
            [self.e_s_real(g, place) for g in G2free], place)
        return max([c5, (2 * (log(2) + c8 * log(c8 / c7))) / c7, 0])

    def initial_bound_complex_case(self, G2free, place, g0, c3):
        r"""
        Return the initial bound of Lemma IX.1.2 at a complex ``place``.
        """
        if self.is_real_place(place):
            raise ValueError('place is not real')
        if g0.multiplicative_order() == Infinity:
            raise ValueError('g0 does not have finite multiplicative order')

        c5 = (2 * log(2)) / c3
        c7 = c3 / 2

        Gtorsion = [g0 ** i for i in range(g0.multiplicative_order())]
        B = 0
        n = len(G2free)
        for ad in Gtorsion:
            if ad != -1:
                c8 = self.Baker_Wustholz_low_lattice_bound(
                    G2free + [(-1) * ad, -1], place)
            else:
                c8 = self.Baker_Wustholz_low_lattice_bound(
                    G2free + [-1], place)
            B = max([c5,
                     (2 * (log(2) + c8 * log(c8 / c7)
                           + c8 * log(n + 1))) / c7,
                     B])
        return B

    def initial_bound_finite_case(self, G2free, prime, g0, c3, embeddings):
        r"""
        Return the initial bound of Lemma IX.1.3 at the finite ``prime``.

        OUTPUT:

        a triple ``(B, M0, M)`` as in Lemma IX.3 of Smart's book.
        """
        p = (prime.norm()).factor()[0][0]
        f = prime.residue_class_degree()
        e = prime.absolute_ramification_index()
        c9 = c3 / (e * f * log(p))

        M0, M, k = self.a_basis_with_0_order_at_p(prime, [g0] + G2free)
        B = 0
        for m0 in M0:
            c11, c12 = self.Yu_theorem([m0] + M, prime, embeddings)
            B = max([B, (2 * (c12 + c11 * log(c11 / (e * c9)))) / (e * c9)])

        return B, M0, M

    # ------------------------------------------------------------------ #
    # reduction (step 2)
    # ------------------------------------------------------------------ #
    def reduction_step_real_case(self, place, B0, G, c7):
        r"""
        Reduce the bound ``B0`` at the real ``place`` using LLL.

        OUTPUT:

        a pair ``(B, increase_precision)``.
        """
        n = len(G)
        Glog = [log(place(self.e_s_real(g, place))) for g in G]
        if len([1 for g in G
                if place(self.e_s_real(g, place)).is_zero()]) > 0:
            return 0, True

        C = self._round(max([1 / l.abs() for l in Glog if l != 0]) + 1)

        if place.codomain().precision() < log(C) / log(2):
            return 0, True

        S = (n - 1) * (B0) ** 2
        T = (1 + n * B0) / 2
        finish = False
        while not finish:
            A = copy(identity_matrix(ZZ, n))
            v = vector([self._round(g * C) for g in Glog])

            if v[n - 1] == 0:
                k = [i for i, a in enumerate(v) if not a.is_zero()][0]
                v[n - 1] = v[k]
                v[k] = 0
            A[n - 1] = v

            A = A.transpose()
            y = copy(zero_vector(ZZ, n))
            l = minimal_vector(A, y)

            if l < T ** 2 + S:
                C = 2 * C
                if place.codomain().precision() < log(C) / log(2):
                    return 0, True
            else:
                if sqrt(l - S) - T > 0:
                    return self._round(
                        (log(C * 2) - log(sqrt(l - S) - T)) / c7), False
                return B0, False

    def reduction_step_complex_case(self, place, B0, G, g0, c7):
        r"""
        Reduce the bound ``B0`` at the complex ``place`` using LLL.

        OUTPUT:

        a pair ``(B, increase_precision)``.
        """
        precision = place.codomain().precision()
        n = len(G)
        Glog_imag = [(log(place(g))).imag_part() for g in G]
        Glog_real = [(log(place(g))).real_part() for g in G]
        Glog_imag = Glog_imag + [2 * pi]
        Glog_real = Glog_real + [0]
        a0log_imag = (log(place(-g0))).imag_part()
        a0log_real = (log(place(-g0))).real_part()

        pl = self.higher_precision(place, 2 * place.codomain().precision())
        if len([g for g in G
                if (pl(g).abs() - 1).abs()
                > 2 ** (-place.codomain().precision())]) == 0:
            C = 1
            S = n * B0 ** 2
            T = ((n + 1) * B0 + 1) / 2
            finish = False
            while not finish:
                A = copy(identity_matrix(ZZ, n + 1))
                v = vector([self._round(g * C) for g in Glog_imag])

                if v[n] == 0:
                    k = [i for i, a in enumerate(v) if not a.is_zero()][0]
                    v[n] = v[k]
                    v[k] = 0
                A[n] = v

                if A.is_singular():
                    C *= 2
                else:
                    A = A.transpose()
                    y = copy(zero_vector(ZZ, n + 1))
                    y[n] = (-1) * self._round(a0log_imag * C)
                    l = minimal_vector(A, y)

                    if l < T ** 2 + S:
                        C = 2 * C
                        if precision < log(C) / log(2):
                            return 0, True
                    else:
                        Bnew = self._round(
                            (log(C * 2) - log(sqrt(l - S) - T)) / c7)
                        finish = True
                        if mod(y[n], A[n, n]) == 0:
                            return max(Bnew, (y[n] / A[n, n]).abs()), False
                        return Bnew, False
        else:
            C = 1
            S = (n - 1) * B0 ** 2
            T = ((n + 1) * B0 + 1) / sqrt(2)
            finish = False

            k = [i for i in range(len(Glog_real)) if Glog_real[i] != 0][0]
            a = Glog_real[k]
            Glog_real[k] = Glog_real[n - 1]
            Glog_real[n - 1] = a

            a = Glog_imag[k]
            Glog_imag[k] = Glog_imag[n - 1]
            Glog_imag[n - 1] = a

            while not finish:
                A = copy(identity_matrix(ZZ, n + 1))
                A[n - 1] = vector([self._round(g * C) for g in Glog_real])
                A[n] = vector([self._round(g * C) for g in Glog_imag])

                if A.is_singular():
                    C *= 2
                else:
                    A = A.transpose()
                    y = copy(zero_vector(ZZ, n + 1))
                    y[n] = (-1) * self._round(a0log_imag * C)
                    y[n - 1] = (-1) * self._round(a0log_real * C)
                    l = minimal_vector(A, y)

                    if l < T ** 2 + S:
                        C *= 2
                        if precision < log(C) / log(2):
                            return 0, True
                    else:
                        Bnew = self._round(
                            (log(C * 2) - log(sqrt(l - S) - T)) / c7)

                        M = matrix(ZZ, 2, [A[n - 1, n - 1], A[n - 1, n],
                                           A[n, n - 1], A[n, n]])
                        b = vector(ZZ, 2, [-y[n - 1], -y[n]])
                        if M.determinant() == 1 or M.determinant() == -1:
                            x = M.inverse() * b
                            return max(Bnew, x[0].abs(), x[1].abs()), False
                        return Bnew, False

    def reduction_step_finite_case(self, prime, B0, M, M_logp, m0, c3,
                                   precision):
        r"""
        Reduce the bound ``B0`` at the finite ``prime`` using LLL.

        OUTPUT:

        a pair ``(B, increase_precision)``.
        """
        if len([g for g in M + [m0] if g.valuation(prime) != 0]) > 0:
            raise ValueError(
                'There is an element with non zero valuation at prime')

        K = prime.ring()
        p = prime.absolute_norm().factor()[0][0]
        f = prime.residue_class_degree()
        e = prime.absolute_ramification_index()
        c5 = c3 / (f * e * log(p))
        theta = K.gen()

        if len(M) == 0:
            if m0 != 1:
                return RR(max(log(p) * f * (m0 - 1).valuation(prime) / c3,
                              0)).floor(), False
            return 0, False

        m0_logp = log_p(m0, prime, precision)
        m0_logp = embedding_to_Kp(m0_logp, prime, precision)
        n = len(M_logp)

        Theta = [theta ** i for i in range(K.absolute_degree())]
        ordp_Disc = (K.disc(Theta)).valuation(p)

        c8 = min([min([a.valuation(p) for a in g]) for g in M_logp])
        l = p ** c8

        low_bound = self._round(1 / c5)
        for a in m0_logp:
            if a != 0:
                if c8 > a.valuation(p):
                    B1 = (c8 + ordp_Disc / 2) / c5
                    if B1 > low_bound:
                        return RR(B1).floor(), False
                    return low_bound, False

        c8 = min([a.valuation(p) for a in m0_logp] + [c8])
        B = [g / l for g in M_logp]
        b0 = m0_logp / l
        c9 = c8 + ordp_Disc / 2

        m = e * f
        u = 1
        finish = False
        while not finish:
            if u > (precision * log(2)) / log(p):
                return 0, True

            A11 = copy(identity_matrix(ZZ, n))
            A12 = copy(zero_matrix(ZZ, n, m))
            A21 = copy(zero_matrix(ZZ, n, m))
            A22 = p ** u * copy(identity_matrix(ZZ, m))
            for i, b in enumerate(B):
                A21[i] = vector([mod(b[j], p ** u) for j in range(m)])
            A = block_matrix([[A11, A12], [A21.transpose(), A22]])

            y = copy(zero_vector(ZZ, n + m))
            for i in range(m):
                y[i + n] = -mod(b0[i], p ** u)

            l = minimal_vector(A.transpose(), y)
            if l > n * B0 ** 2:
                B2 = (u + c9) / c5
                if B2 > low_bound:
                    return RR(B2).floor(), False
                return low_bound, False
            u += 1

    # ------------------------------------------------------------------ #
    # torsion and the bound
    # ------------------------------------------------------------------ #
    def _torsion_generator(self, tors):
        tors = [z for z in tors if z != 0]
        if len(tors) == 0:
            return self.K(1)
        return max(tors, key=lambda z: z.multiplicative_order())

    def initial_bound(self):
        r"""
        Return the initial upper bound for the exponents of the solutions
        (step 1 of the algorithm).
        """
        if self._initial_bound is None:
            self._initial_bound()
        return self._initial_bound

    def reduce(self, B1=None):
        r"""
        Return the reduced upper bound for the exponents of the solutions
        (step 2 of the algorithm).

        INPUT:

        - ``B1`` -- an initial upper bound.  If ``None`` (default),
          :meth:`initial_bound` is used.
        """
        if B1 is None:
            B1 = self.initial_bound()
        elif self.supports is None:
            self._initial_bound()
        self._reduce_bound(B1)
        return self._B

    def bound(self):
        r"""
        Return the (reduced) upper bound for the exponents of the solutions,
        i.e. the result of applying :meth:`reduce` to :meth:`initial_bound`.
        """
        initial_bound = self._initial_bound()
        if self._B is None:
            self._reduce_bound(initial_bound)
        return self._B

    def _initial_bound(self):
        r"""
        Compute and cache step 1: the initial upper bound `B_1`.
        """

        if len(self._G1free) == 0 or len(self._G2free) == 0:
            self._initial_bound = 0
            self._B = 0
            return

        finiteSup1, realSup1, complexSup1 = self.support_of_G(self._G1all)
        finiteSup2, realSup2, complexSup2 = self.support_of_G(self._G2all)
        self.supports = ((finiteSup1, realSup1, complexSup1),
                         (finiteSup2, realSup2, complexSup2))

        G1c1, G1c2, G1c3 = self.c_constants(self._G1free, self.prec)
        G2c1, G2c2, G2c3 = self.c_constants(self._G2free, self.prec)
        self.c = ((G1c1, G1c2, G1c3), (G2c1, G2c2, G2c3))

        initial_bound_real = max([self.initial_bound_real_case(self._G2free, p, G1c3)
                      for p in realSup1]
                     + [self.initial_bound_real_case(self._G1free, p, G2c3)
                        for p in realSup2]
                     + [0])

        initial_bound_complex = max([self.initial_bound_complex_case(
            self._G2free, p, self._g02, G1c3) for p in complexSup1]
                        + [self.initial_bound_complex_case(
                            self._G1free, p, self._g01, G2c3)
                           for p in complexSup2]
                        + [0])

        G1finite_init = []
        initla_bound_finite_G1 = 0
        for prime in finiteSup1:
            B1, M0, M = self.initial_bound_finite_case(
                self._G2free, prime, self._g02, G1c3, self._Kembeddings)
            G1finite_init.append([prime, M0, M])
            initla_bound_finite_G1 = max(initla_bound_finite_G1, B1)

        G2finite_init = []
        initla_bound_finite_G2 = 0
        for prime in finiteSup2:
            B1, M0, M = self.initial_bound_finite_case(
                self._G1free, prime, self._g01, G2c3, self._Kembeddings)
            G2finite_init.append([prime, M0, M])
            initla_bound_finite_G2 = max(initla_bound_finite_G2, B1)
        initial_bound_finite = max(initla_bound_finite_G1, initla_bound_finite_G2)

        self._finite_init = (G1finite_init, G2finite_init)
        self._initial_bound = self._RR(max(initial_bound_real, initial_bound_complex, initial_bound_finite)).floor()
        if self.verbose:
            print("initial bound: %s" % self._initial_bound)

        return self._RR(max(initial_bound_real, initial_bound_complex, initial_bound_finite)).floor()

    def _reduce_bound(self):
        r"""
        Compute and cache step 2: the reduced upper bound, starting from the
        initial bound ``B1``.
        """
        same = self._same_groups(self._G1all, self._G2all)

        reduced_bound_real_G1 = self._reduce_real(self.supports[0][1], self._G2free,
                                   self.c[0][2], self._G1free, self._reduced_bound_G1)
        reduced_bound_complex_G1 = self._reduce_complex(self.supports[0][2], self._G2free,
                                         self._g02, self.c[0][2], self._G1free,
                                         self._reduced_bound_G1)
        reduced_bound_finite_G1 = self._reduce_finite(self._finite_init[0], self._G1free,
                                       self.c[0][2], self._reduced_bound_G1)

        self._reduced_bound_G1 = self._RR(
            max(reduced_bound_real_G1, reduced_bound_complex_G1, reduced_bound_finite_G1)
        ).floor()


        if not same:
            reduced_bound_real_G2 = self._reduce_real(self.supports[1][1], self._G1free, elf.c[1][2], self._G2free,
                                                      self._reduced_bound_G2)
            reduced_bound_complex_G2 = self._reduce_complex(self.supports[1][2], self._G1free, self._g01, self.c[1][2],
                                                            self._G2free, self._reduced_bound_G2)
            reduced_bound_finite_G2 = self._reduce_finite(self._finite_init[1], self._G2free, self.c[1][2],
                                                          self._reduced_bound_G2)
            self._reduced_bound_G2 = self._RR(
                max(reduced_bound_real_G2, reduced_bound_complex_G2, reduced_bound_finite_G2)
            ).floor()

    def _reduce_real(self, places, other_free, c3, self_free, B0):
        r"""
        Reduce the bound at a list of real places.

        ``c3`` is the constant of the group whose exponents are being
        bounded (with free generators ``self_free``), and ``other_free`` are
        the free generators of the other group.
        """
        B_side = 0
        for place in places:
            Bold = B0
            finish = False
            while not finish:
                Bnew, increase_precision = self.reduction_step_real_case(
                    place, Bold, other_free, c3)
                if not increase_precision:
                    if Bnew < Bold:
                        Bold = Bnew
                        if Bold <= 2:
                            finish = True
                            Bold = 2
                    else:
                        finish = True
                else:
                    c3 = self.c_constants(
                        self_free, 2 * place.codomain().precision())[2]
                    place = self.higher_precision(
                        place, 2 * place.codomain().precision())
            B_side = max(B_side, Bold)
        return B_side

    def _reduce_complex(self, places, other_free, other_torsion, c3,
                        self_free, B0):
        r"""
        Reduce the bound at a list of complex places.
        """
        B_side = 0
        for place in places:
            B_place = 0
            for g0 in [other_torsion ** i
                       for i in range(other_torsion.multiplicative_order())]:
                Bold = B0
                finish = False
                while not finish:
                    Bnew, increase_precision = \
                        self.reduction_step_complex_case(
                            place, Bold, other_free, g0, c3 / 2)
                    if not increase_precision:
                        if Bnew < Bold:
                            Bold = Bnew
                            if Bold <= 2:
                                finish = True
                                Bold = 2
                        else:
                            finish = True
                    else:
                        c3 = self.c_constants(
                            self_free, 2 * place.codomain().precision())[2]
                        place = self.higher_precision(
                            place, 2 * place.codomain().precision())
                B_place = max(B_place, Bold)
            B_side = max(B_side, B_place)
        return B_side

    def _reduce_finite(self, finite_init, self_free, c3, B0):
        r"""
        Reduce the bound at the finite places using the precomputed data
        ``finite_init`` (a list of ``[prime, M0, M]`` triples).
        """
        B_side = 0
        for P in finite_init:
            B_place = 0
            if len(P[2]) != 0:
                prec = self.prec
                M_logp = [embedding_to_Kp(log_p(m, P[0], prec), P[0], prec)
                          for m in P[2]]
                for m0 in P[1]:
                    Bold = B0
                    finish = False
                    while not finish:
                        Bnew, increase_precision = \
                            self.reduction_step_finite_case(
                                P[0], Bold, P[2], M_logp, m0, c3, prec)
                        if not increase_precision:
                            if Bnew < Bold:
                                Bold = Bnew
                                if Bold <= 2:
                                    finish = True
                                    Bold = 2
                            else:
                                finish = True
                        else:
                            prec *= 2
                            c3 = self.c_constants(self_free, prec)[2]
                            M_logp = [embedding_to_Kp(
                                log_p(m, P[0], prec), P[0], prec)
                                for m in P[2]]
                    B_place = max(B_place, Bold)
            B_side = max(B_side, B_place)
        return B_side

    # ------------------------------------------------------------------ #
    # enumeration (step 3)
    # ------------------------------------------------------------------ #
    def _in_G2(self, y):
        r"""
        Return ``True`` if ``y`` lies in `G_2`.
        """
        if y == 0:
            return False
        if len(self._G2all) == 0:
            return y == 1
        if self._G2_ctx is None:
            self._G2_ctx = self._group_context(self._G2all)
        SunitK, A, supp = self._G2_ctx
        if not self.K.ideal(y).is_S_unit(S=supp):
            return False
        return vector(SunitK(y).list()) in A.row_space()

    def _simple_loop(self, B):
        r"""
        Enumerate the solutions `x` of `x + y = 1` with the exponents of
        `x` in `G_1` bounded by ``B`` in absolute value.
        """
        solutions = []
        r = len(self._G1free)
        for zeta in self._torsion1:
            for e in _iproduct(*([range(-B, B + 1)] * r)):
                x = zeta * prod(g ** ei for g, ei in zip(self._G1free, e))
                if self._in_G2(1 - x):
                    if x not in solutions:
                        solutions.append(x)
        return solutions

    def solve(self):
        r"""
        Return the list of all `x` in `G_1` such that `1 - x` lies in `G_2`.

        EXAMPLES::

            sage: import os, sys
            sage: sys.path.insert(0, os.getcwd())
            sage: from s_unit_equation import SUnitSolver
            sage: K.<a> = NumberField(x^2 + 5)
            sage: S = K.primes_above(30)
            sage: G = [g for g in K.S_unit_group(S=S).gens_values()
            ....:      if g.multiplicative_order() == Infinity]
            sage: sols = SUnitSolver(G, G).solve()
            sage: all(x + y == 1 for x, y in SUnitSolver(G, G).solve_pairs())
            True
        """
        B = self.bound()
        return self._simple_loop(B)

    def solve_pairs(self):
        r"""
        Return the list of pairs ``(x, 1 - x)`` solving the equation.
        """
        return [(x, 1 - x) for x in self.solve()]
