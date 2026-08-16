import logging

import numpy as np

from goppa.goppacodegenerator import GoppaCodeGenerator
from mceliece.mathutils import (
    GF2Matrix,
    GF2Poly,
    GF2m,
    GF2mPoly,
    GF2mRing,
    ext_euclid_poly_alt,
    get_binary_from_alpha,
    random_inv_matrix,
)


log = logging.getLogger("mceliececipher")


class McElieceCipher:
    """
    Binary Goppa-code McEliece implementation.

    Parameters
    ----------
    m : int
        Extension degree. The Goppa code is defined using GF(2^m).

    n : int
        Code length.

    t : int
        Degree of the Goppa polynomial and the number of errors
        injected by the baseline McEliece encryption algorithm.

    Notes
    -----
    The implementation uses

        G       secret binary Goppa generator matrix
        H       secret binary parity-check matrix
        R       right inverse of G
        S       secret invertible scrambling matrix
        S_inv   inverse scrambling matrix
        P       secret permutation vector
        P_inv   inverse permutation vector
        Gp      public generator matrix

    satisfying

        G H^T = 0,

        G R = I_k,

        Gp = S G P.

    Phase 2B represents the coordinate permutation using integer
    permutation vectors instead of dense n x n permutation matrices.

    For a row vector x,

        x P := x[P].

    Therefore

        (x P) P^{-1}
        = x[P][P_inv]
        = x.

    Phase 2C introduces a precomputed right inverse

        R in GF(2)^{n x k}

    satisfying

        G R = I_k.

    After Patterson decoding produces a clean codeword

        c_clean = a G,

    the information vector is recovered directly as

        a = c_clean R.

    Hence normal decoding does not solve a new linear system for every
    ciphertext.
    """

    # ============================================================
    # Construction
    # ============================================================

    def __init__(
        self,
        m,
        n,
        t,
    ):
        self.m = int(m)
        self.n = int(n)
        self.t = int(t)

        self.q = 2

        # --------------------------------------------------------
        # Parameter validation
        # --------------------------------------------------------

        if self.m <= 0:
            raise ValueError(
                "m must be positive."
            )

        if self.n <= 0:
            raise ValueError(
                "n must be positive."
            )

        if self.t <= 0:
            raise ValueError(
                "t must be positive."
            )

        if self.n > 2 ** self.m:
            raise ValueError(
                f"n={self.n} exceeds "
                f"field size 2^m={2 ** self.m}."
            )

        log.info(
            "McEliece("
            f"m={self.m}, "
            f"n={self.n}, "
            f"t={self.t}, "
            f"q={self.q}, "
            f"q^m={self.q ** self.m}"
            ") initiated"
        )

        # --------------------------------------------------------
        # Secret Goppa-code data
        # --------------------------------------------------------

        self.G = None
        self.H = None
        self.k = None

        # --------------------------------------------------------
        # Phase 2C right inverse
        #
        #     G R = I_k.
        #
        # R has shape n x k.
        # --------------------------------------------------------

        self.R = None

        # Indices of the k independent columns of G used to
        # construct R.
        self.right_inverse_pivots = None

        # --------------------------------------------------------
        # McEliece permutation
        #
        # Phase 2B representation:
        #
        #     P     : ndarray shape (n,)
        #     P_inv : ndarray shape (n,)
        # --------------------------------------------------------

        self.P = None
        self.P_inv = None

        # --------------------------------------------------------
        # Scrambling matrix
        # --------------------------------------------------------

        self.S = None
        self.S_inv = None

        # --------------------------------------------------------
        # Public generator
        # --------------------------------------------------------

        self.Gp = None

        # --------------------------------------------------------
        # Goppa polynomial and extension-field modulus
        # --------------------------------------------------------

        self.g_poly = None
        self.irr_poly = None

    # ============================================================
    # Internal conversion helpers
    # ============================================================

    @staticmethod
    def _as_gf2_matrix(value):
        """
        Convert value into GF2Matrix if necessary.

        This allows the class to work with objects generated directly
        in memory and primitive NumPy arrays restored from serialized
        keys.
        """

        if isinstance(
            value,
            GF2Matrix,
        ):
            return value

        arr = np.asarray(
            value
        )

        # A NumPy object array may contain a single GF2Matrix object
        # when reading legacy serialized data.
        if arr.ndim == 0:
            item = arr.item()

            if isinstance(
                item,
                GF2Matrix,
            ):
                return item

            arr = np.asarray(
                item
            )

        return GF2Matrix.from_list(
            arr
        )

    @staticmethod
    def _as_permutation_vector(
        value,
        n,
        name="permutation",
    ):
        """
        Normalize and validate a permutation vector.

        A valid permutation contains every integer

            0, 1, ..., n-1

        exactly once.
        """

        arr = np.asarray(
            value,
            dtype=np.int64,
        ).reshape(-1)

        if arr.shape != (
            n,
        ):
            raise ValueError(
                f"{name} must have shape ({n},), "
                f"received {arr.shape}."
            )

        expected = np.arange(
            n,
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(arr),
            expected,
        ):
            raise ValueError(
                f"{name} is not a valid "
                f"permutation of 0,...,{n - 1}."
            )

        return arr

    @staticmethod
    def _gf2_matrix_equal(
        A,
        B,
    ):
        """
        Compare two GF2Matrix objects entry by entry.
        """

        A = McElieceCipher._as_gf2_matrix(
            A
        )

        B = McElieceCipher._as_gf2_matrix(
            B
        )

        if A.arr.shape != B.arr.shape:
            return False

        return all(
            int(a.n) == int(b.n)
            for a, b in zip(
                A.arr.flat,
                B.arr.flat,
            )
        )

    # ============================================================
    # Permutation helpers
    # ============================================================

    @staticmethod
    def _permute_vector(
        vector,
        permutation,
    ):
        """
        Apply a permutation to a GF(2) row vector.

        If

            y = x[P],

        this method returns y.
        """

        vector = (
            McElieceCipher._as_gf2_matrix(
                vector
            )
        )

        permutation = np.asarray(
            permutation,
            dtype=np.int64,
        ).reshape(-1)

        if vector.arr.ndim != 1:
            raise ValueError(
                "_permute_vector expects "
                "a one-dimensional GF2 vector."
            )

        if len(vector) != len(
            permutation
        ):
            raise ValueError(
                "Vector and permutation lengths "
                "do not agree."
            )

        return GF2Matrix(
            vector.arr[
                permutation
            ]
        )

    @staticmethod
    def _permute_columns(
        matrix,
        permutation,
    ):
        """
        Apply a right-hand coordinate permutation to matrix columns.

        For A in GF(2)^{k x n}, this implements

            A P

        as

            A[:, P].
        """

        matrix = (
            McElieceCipher._as_gf2_matrix(
                matrix
            )
        )

        permutation = np.asarray(
            permutation,
            dtype=np.int64,
        ).reshape(-1)

        if matrix.arr.ndim != 2:
            raise ValueError(
                "_permute_columns expects "
                "a two-dimensional matrix."
            )

        if matrix.arr.shape[1] != len(
            permutation
        ):
            raise ValueError(
                "Matrix column count and "
                "permutation length do not agree."
            )

        return GF2Matrix(
            matrix.arr[
                :,
                permutation
            ]
        )

    # ============================================================
    # Phase 2C right-inverse construction
    # ============================================================

    def _construct_right_inverse(self):
        """
        Construct a right inverse R of the generator matrix G.

        Let

            G in GF(2)^{k x n}

        have full row rank k.

        Select k independent columns with indices

            J = (j_1, ..., j_k)

        and define

            B = G[:, J].

        Since B is invertible, define

            R[J, :] = B^{-1}

        and set every other row of R equal to zero.

        Consequently

            G R
              = G[:, J] B^{-1}
              = B B^{-1}
              = I_k.

        Returns
        -------
        R : GF2Matrix
            Right inverse with shape (n, k).

        pivots : numpy.ndarray
            Integer vector containing the k selected independent
            column indices.
        """

        if self.G is None:
            raise RuntimeError(
                "Generator matrix G is unavailable."
            )

        G = self._as_gf2_matrix(
            self.G
        )

        if G.arr.ndim != 2:
            raise ValueError(
                "Generator matrix G must be "
                "two-dimensional."
            )

        k, n = G.arr.shape

        if k <= 0:
            raise ValueError(
                "Generator matrix must have "
                "positive row dimension."
            )

        if n <= 0:
            raise ValueError(
                "Generator matrix must have "
                "positive column dimension."
            )

        if k > n:
            raise ValueError(
                "Generator matrix cannot have more "
                "rows than columns."
            )

        # --------------------------------------------------------
        # 1. Compute RREF of G once.
        #
        # Row operations preserve column dependence relations.
        # Since rank(G)=k, exactly k pivot columns exist.
        # --------------------------------------------------------

        G_rref, rank = (
            G.rref()
        )

        if rank != k:
            raise RuntimeError(
                f"Generator matrix has rank {rank}; "
                f"expected full row rank {k}."
            )

        # --------------------------------------------------------
        # 2. Extract pivot-column positions.
        # --------------------------------------------------------

        pivots = []

        for row_index in range(
            k
        ):
            pivot = None

            for column_index in range(
                n
            ):
                value = int(
                    G_rref.arr[
                        row_index,
                        column_index,
                    ].n
                )

                if value == 1:
                    pivot = column_index
                    break

            if pivot is None:
                raise RuntimeError(
                    "Unable to identify a pivot "
                    f"for RREF row {row_index}."
                )

            pivots.append(
                pivot
            )

        pivots = np.asarray(
            pivots,
            dtype=np.int64,
        )

        if pivots.shape != (
            k,
        ):
            raise RuntimeError(
                "Unexpected pivot-vector shape."
            )

        if len(
            np.unique(
                pivots
            )
        ) != k:
            raise RuntimeError(
                "Pivot-column extraction produced "
                "duplicate indices."
            )

        if (
            np.any(
                pivots < 0
            )
            or
            np.any(
                pivots >= n
            )
        ):
            raise RuntimeError(
                "Pivot-column extraction produced "
                "an out-of-range index."
            )

        # --------------------------------------------------------
        # 3. Form
        #
        #     B = G[:, J].
        # --------------------------------------------------------

        B = GF2Matrix(
            G.arr[
                :,
                pivots
            ]
        )

        if B.arr.shape != (
            k,
            k,
        ):
            raise RuntimeError(
                f"Pivot submatrix has shape "
                f"{B.arr.shape}; "
                f"expected ({k}, {k})."
            )

        _, B_rank = (
            B.rref()
        )

        if B_rank != k:
            raise RuntimeError(
                "Selected pivot submatrix is "
                "not invertible over GF(2)."
            )

        # --------------------------------------------------------
        # 4. Exact inverse over GF(2)
        # --------------------------------------------------------

        B_inv = (
            B.inv()
        )

        if B_inv.arr.shape != (
            k,
            k,
        ):
            raise RuntimeError(
                "Inverse pivot submatrix has "
                "unexpected dimensions."
            )

        # --------------------------------------------------------
        # 5. Construct R
        #
        #     R[J, :] = B^{-1}.
        # --------------------------------------------------------

        B_inv_numpy = np.array(
            [
                int(entry.n)
                for entry in B_inv.arr.flat
            ],
            dtype=np.uint8,
        ).reshape(
            k,
            k,
        )

        R_numpy = np.zeros(
            (
                n,
                k,
            ),
            dtype=np.uint8,
        )

        R_numpy[
            pivots,
            :
        ] = B_inv_numpy

        R = GF2Matrix.from_list(
            R_numpy
        )

        if R.arr.shape != (
            n,
            k,
        ):
            raise RuntimeError(
                f"Right inverse has shape "
                f"{R.arr.shape}; "
                f"expected ({n}, {k})."
            )

        # --------------------------------------------------------
        # 6. Verify G R = I_k once during construction.
        # --------------------------------------------------------

        GR = (
            G
            * R
        )

        identity = GF2Matrix.from_list(
            np.eye(
                k,
                dtype=np.uint8,
            )
        )

        if not self._gf2_matrix_equal(
            GR,
            identity,
        ):
            raise RuntimeError(
                "Right-inverse construction failed: "
                "G R != I_k."
            )

        log.info(
            "Constructed generator right inverse "
            f"R with shape {R.arr.shape}."
        )

        return (
            R,
            pivots,
        )

    # ============================================================
    # Key generation
    # ============================================================

    def generate_random_keys(self):
        """
        Generate a binary Goppa code and McEliece key pair.

        The secret code objects satisfy

            G H^T = 0,

            G R = I_k.

        The public generator satisfies

            G_pub = S G P.

        P is represented by a permutation vector.
        """

        # --------------------------------------------------------
        # 1. Generate binary Goppa code
        # --------------------------------------------------------

        (
            self.G,
            self.H,
            self.g_poly,
            self.irr_poly,
        ) = GoppaCodeGenerator(
            self.m,
            self.n,
            self.t,
        ).gen()

        # --------------------------------------------------------
        # 2. Store Goppa polynomial coefficients.
        #
        # Each coefficient in GF(2^m) is represented by its
        # m-bit coefficient vector in the basis
        #
        #     1, alpha, ..., alpha^(m-1).
        # --------------------------------------------------------

        g_coeffs = []

        for coefficient in (
            self.g_poly
            .all_coeffs()[::-1]
        ):
            bits = get_binary_from_alpha(
                coefficient,
                self.irr_poly,
                2,
            )

            g_coeffs.append(
                [
                    int(bit.n)
                    for bit in bits
                ]
            )

        self.g_poly = np.array(
            g_coeffs,
            dtype=np.uint8,
        )

        # --------------------------------------------------------
        # 3. Store irreducible field polynomial
        # --------------------------------------------------------

        self.irr_poly = np.array(
            [
                int(c) % 2
                for c
                in self.irr_poly
                .all_coeffs()[::-1]
            ],
            dtype=np.uint8,
        )

        # --------------------------------------------------------
        # 4. Determine actual binary code dimension
        # --------------------------------------------------------

        self.k = (
            self.G.arr.shape[0]
        )

        log.info(
            "Generated binary Goppa code "
            f"[n={self.n}, k={self.k}]"
        )

        # --------------------------------------------------------
        # 5. Phase 2C: construct right inverse
        #
        #     G R = I_k.
        # --------------------------------------------------------

        (
            self.R,
            self.right_inverse_pivots,
        ) = self._construct_right_inverse()

        if self.R.arr.shape != (
            self.n,
            self.k,
        ):
            raise RuntimeError(
                "Right inverse has unexpected "
                f"shape {self.R.arr.shape}; "
                f"expected "
                f"({self.n}, {self.k})."
            )

        # --------------------------------------------------------
        # 6. Generate secret permutation
        # --------------------------------------------------------

        self.P = np.random.permutation(
            self.n
        ).astype(
            np.int64
        )

        self.P_inv = np.argsort(
            self.P
        ).astype(
            np.int64
        )

        # --------------------------------------------------------
        # 7. Validate permutation and inverse
        # --------------------------------------------------------

        identity_positions = np.arange(
            self.n,
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(
                self.P
            ),
            identity_positions,
        ):
            raise RuntimeError(
                "Generated P is not a valid "
                "permutation."
            )

        if not np.array_equal(
            np.sort(
                self.P_inv
            ),
            identity_positions,
        ):
            raise RuntimeError(
                "Generated P_inv is not a valid "
                "permutation."
            )

        if not np.array_equal(
            self.P[
                self.P_inv
            ],
            identity_positions,
        ):
            raise RuntimeError(
                "Invalid permutation inverse: "
                "P[P_inv] != identity."
            )

        if not np.array_equal(
            self.P_inv[
                self.P
            ],
            identity_positions,
        ):
            raise RuntimeError(
                "Invalid permutation inverse: "
                "P_inv[P] != identity."
            )

        # --------------------------------------------------------
        # 8. Generate secret scrambling matrix
        # --------------------------------------------------------

        self.S = GF2Matrix.from_list(
            random_inv_matrix(
                self.k
            )
        )

        self.S_inv = (
            self.S.inv()
        )

        # --------------------------------------------------------
        # 9. Construct public generator
        #
        #     G_pub = S G P.
        #
        # With vector P:
        #
        #     G_pub = (S G)[:, P].
        # --------------------------------------------------------

        SG = (
            self.S
            * self.G
        )

        self.Gp = (
            self._permute_columns(
                SG,
                self.P,
            )
        )

        # --------------------------------------------------------
        # 10. Final key-generation sanity checks
        # --------------------------------------------------------

        if self.Gp.arr.shape != (
            self.k,
            self.n,
        ):
            raise RuntimeError(
                "Public generator has unexpected "
                f"shape {self.Gp.arr.shape}."
            )

        # R was already verified during construction. The following
        # check remains in key generation, where its cost is paid
        # only once per key pair.

        GR = (
            self.G
            * self.R
        )

        I_k = GF2Matrix.from_list(
            np.eye(
                self.k,
                dtype=np.uint8,
            )
        )

        if not self._gf2_matrix_equal(
            GR,
            I_k,
        ):
            raise RuntimeError(
                "Key-generation sanity check "
                "failed: G R != I_k."
            )

        log.info(
            "McEliece key generation completed."
        )

    # ============================================================
    # Encryption
    # ============================================================

    def encrypt(
        self,
        msg_arr,
    ):
        """
        Encrypt a k-bit message using the public generator.

        Baseline McEliece encryption computes

            c = m G_pub + e,

        where

            wt(e) = t.
        """

        if self.Gp is None:
            raise RuntimeError(
                "Public key has not been "
                "generated or loaded."
            )

        Gp = self._as_gf2_matrix(
            self.Gp
        )

        k = Gp.arr.shape[0]
        n = Gp.arr.shape[1]

        msg_arr = np.asarray(
            msg_arr,
            dtype=np.uint8,
        ).flatten()

        if len(msg_arr) != k:
            raise ValueError(
                "Wrong message length. "
                f"Expected {k} bits, "
                f"received {len(msg_arr)}."
            )

        if self.t > n:
            raise ValueError(
                f"Cannot insert {self.t} errors "
                f"into a ciphertext of length {n}."
            )

        if not np.all(
            (msg_arr == 0)
            | (msg_arr == 1)
        ):
            raise ValueError(
                "Plaintext must be binary."
            )

        log.debug(
            f"plaintext = {msg_arr}"
        )

        message = GF2Matrix.from_list(
            msg_arr
        )

        # --------------------------------------------------------
        # Clean public codeword
        #
        #     c_0 = m G_pub.
        # --------------------------------------------------------

        ciphertext = (
            message
            * Gp
        )

        log.debug(
            f"clean ciphertext = {ciphertext}"
        )

        # --------------------------------------------------------
        # Sample exactly t distinct error coordinates
        # --------------------------------------------------------

        bits_to_flip = np.random.choice(
            n,
            size=self.t,
            replace=False,
        )

        log.debug(
            "error positions = "
            f"{bits_to_flip}"
        )

        for position in (
            bits_to_flip
        ):
            ciphertext[
                position
            ] = ciphertext[
                position
            ].flip()

        log.debug(
            f"noisy ciphertext = {ciphertext}"
        )

        return ciphertext

    # ============================================================
    # Patterson error repair
    # ============================================================

    def repair_errors(
        self,
        msg_arr,
        syndrome,
    ):
        """
        Repair errors using the Patterson-style decoder.

        The current Goppa support is

            L_i = alpha^i,

        for

            i = 0,...,n-1.

        Consequently the locator polynomial is evaluated at

            1, alpha, alpha^2, ...
        """

        # --------------------------------------------------------
        # 1. Reconstruct binary extension-field modulus
        # --------------------------------------------------------

        if not isinstance(
            self.irr_poly,
            GF2Poly,
        ):
            self.irr_poly = (
                GF2Poly.from_numpy(
                    np.asarray(
                        self.irr_poly,
                        dtype=np.uint8,
                    )
                )
            )

        ring = GF2mRing(
            self.m,
            self.irr_poly,
        )

        # --------------------------------------------------------
        # 2. Reconstruct Goppa polynomial over GF(2^m)
        # --------------------------------------------------------

        if not isinstance(
            self.g_poly,
            GF2mPoly,
        ):
            coefficients = []

            for coefficient in (
                self.g_poly
            ):
                coefficient_poly = (
                    GF2Poly.from_list(
                        [
                            int(e)
                            for e
                            in coefficient
                        ]
                    )
                )

                coefficients.append(
                    GF2m(
                        coefficient_poly,
                        ring,
                    )
                )

            self.g_poly = (
                GF2mPoly.from_list(
                    coefficients
                )
            )

        log.debug(
            f"irr_poly = {self.irr_poly}"
        )

        log.debug(
            f"g_poly = {self.g_poly}"
        )

        # --------------------------------------------------------
        # 3. Convert binary syndrome to polynomial over GF(2^m)
        # --------------------------------------------------------

        syndrome_matrix = (
            self._as_gf2_matrix(
                syndrome
            )
        )

        syndrome_flat = list(
            syndrome_matrix
            .arr
            .flat
        )

        syndrome_coefficients = []

        number_of_coefficients = (
            len(syndrome_flat)
            // self.m
        )

        for i in range(
            number_of_coefficients
        ):
            block = syndrome_flat[
                i * self.m:
                (i + 1) * self.m
            ]

            binary_poly = (
                GF2Poly.from_list(
                    [
                        int(e)
                        for e
                        in block
                    ]
                )
            )

            syndrome_coefficients.append(
                GF2m(
                    binary_poly,
                    ring,
                )
            )

        S_poly = (
            GF2mPoly.from_list(
                syndrome_coefficients
            )
        )

        log.debug(
            f"S_poly = {S_poly}"
        )

        # --------------------------------------------------------
        # 4. Patterson decoding
        # --------------------------------------------------------

        S_inv_poly = (
            S_poly.inv_mod(
                self.g_poly
            )
        )

        log.debug(
            f"S_inv_poly = {S_inv_poly}"
        )

        log.debug(
            "S_poly * S_inv_poly mod g = "
            f"{(S_poly * S_inv_poly) % self.g_poly}"
        )

        # --------------------------------------------------------
        # Special low-degree case
        # --------------------------------------------------------

        if (
            S_inv_poly.degree() == 1
            and
            S_inv_poly[1]
            .n
            .degree() == 0
            and
            S_inv_poly[1]
            .n
            .poly
            .coeffs()[0] == 1
        ):
            tau_poly = (
                S_inv_poly
            )

        else:
            # ----------------------------------------------------
            # Split g into even/odd components
            # ----------------------------------------------------

            g0, g1 = (
                self.g_poly.split()
            )

            log.debug(
                f"g0 = {g0}"
            )

            log.debug(
                f"g1 = {g1}"
            )

            log.debug(
                "g0^2 + z g1^2 = "
                f"{g0 ** 2 + GF2mPoly.x(ring) * g1 ** 2}"
            )

            g1_inv = (
                g1.inv_mod(
                    self.g_poly
                )
            )

            log.debug(
                f"g1_inv = {g1_inv}"
            )

            w = (
                g0
                * g1_inv
            )

            log.debug(
                f"w = {w}"
            )

            # ----------------------------------------------------
            # H(z) = S^{-1}(z) + z
            # ----------------------------------------------------

            H_poly = (
                S_inv_poly
                +
                GF2mPoly.from_list(
                    [
                        GF2m(
                            GF2Poly.from_list(
                                [0]
                            ),
                            ring,
                        ),
                        GF2m(
                            GF2Poly.from_list(
                                [1]
                            ),
                            ring,
                        ),
                    ]
                )
            )

            log.debug(
                f"H_poly = {H_poly}"
            )

            H0, H1 = (
                H_poly.split()
            )

            log.debug(
                f"H0 = {H0}"
            )

            log.debug(
                f"H1 = {H1}"
            )

            # ----------------------------------------------------
            # Patterson auxiliary polynomial.
            #
            # Use R_poly to distinguish it from self.R, which is
            # the generator right inverse introduced in Phase 2C.
            # ----------------------------------------------------

            R_poly = (
                H0
                +
                w * H1
            )

            log.debug(
                f"R_poly = {R_poly}"
            )

            log.debug(
                "R_poly^2 mod g = "
                f"{(R_poly ** 2) % self.g_poly}"
            )

            b, _, a = (
                ext_euclid_poly_alt(
                    R_poly,
                    self.g_poly,
                    ring,
                    self.t,
                )
            )

            log.debug(
                f"a = {a}"
            )

            log.debug(
                f"b = {b}"
            )

            log.debug(
                f"a^2 = {a ** 2}"
            )

            log.debug(
                f"b^2 = {b ** 2}"
            )

            log.debug(
                "z b^2 = "
                f"{GF2mPoly.x(ring) * b ** 2}"
            )

            # ----------------------------------------------------
            # Error-locator polynomial
            # ----------------------------------------------------

            tau_poly = (
                a ** 2
                +
                GF2mPoly.x(
                    ring
                )
                * b ** 2
            )

        log.debug(
            "error locator tau = "
            f"{tau_poly}"
        )

        # --------------------------------------------------------
        # 5. Evaluate locator polynomial on current support
        #
        #     1, alpha, alpha^2, ...
        # --------------------------------------------------------

        test_elem = ring.one()

        for i in range(
            len(msg_arr)
        ):
            value = (
                tau_poly.eval(
                    test_elem
                )
            )

            log.debug(
                f"tau(alpha^{i}) = {value}"
            )

            if value == 0:
                msg_arr[
                    i
                ] = msg_arr[
                    i
                ].flip()

                log.info(
                    "REPAIRED ERROR "
                    f"ON POSITION {i}"
                )

            test_elem = (
                test_elem
                * ring.alpha()
            )

        return msg_arr

    # ============================================================
    # Goppa decoding
    # ============================================================

    def decode(
        self,
        msg_arr,
    ):
        """
        Decode a secret-coordinate noisy Goppa codeword.

        The decoder computes

            s = c H^T.

        If

            s != 0,

        Patterson correction is applied.

        After correction,

            c_clean = a G.

        Phase 2C recovers a directly using

            a = c_clean R,

        where

            G R = I_k.

        The right inverse is constructed and verified during key
        generation. No G R verification or reconstruction check is
        performed for every ciphertext.
        """

        if self.G is None:
            raise RuntimeError(
                "Secret generator matrix G "
                "is unavailable."
            )

        if self.H is None:
            raise RuntimeError(
                "Parity-check matrix H "
                "is unavailable."
            )

        G = self._as_gf2_matrix(
            self.G
        )

        H = self._as_gf2_matrix(
            self.H
        )

        msg_arr = (
            self._as_gf2_matrix(
                msg_arr
            )
        )

        n = G.arr.shape[1]
        k = G.arr.shape[0]

        if len(msg_arr) != n:
            raise ValueError(
                "Received-word length does "
                "not match the Goppa code length."
            )

        log.debug(
            "received word length = "
            f"{len(msg_arr)}"
        )

        # --------------------------------------------------------
        # 1. Syndrome
        #
        #     s = c H^T.
        # --------------------------------------------------------

        syndrome = (
            msg_arr
            * H.T()
        )

        log.info(
            f"syndrome =\n{syndrome}"
        )

        syndrome_is_zero = all(
            int(x.n) == 0
            for x
            in syndrome.arr.flat
        )

        # --------------------------------------------------------
        # 2. Patterson correction
        # --------------------------------------------------------

        if not syndrome_is_zero:
            msg_arr = (
                self.repair_errors(
                    msg_arr,
                    syndrome,
                )
            )

            # ----------------------------------------------------
            # Verify that Patterson produced a codeword.
            #
            # This check remains useful because it verifies the
            # error-correction result itself.
            # ----------------------------------------------------

            repaired_syndrome = (
                msg_arr
                * H.T()
            )

            repaired_is_zero = all(
                int(x.n) == 0
                for x
                in repaired_syndrome
                .arr
                .flat
            )

            if not repaired_is_zero:
                raise ValueError(
                    "Patterson decoding did not "
                    "produce a valid codeword."
                )

        # --------------------------------------------------------
        # 3. Phase 2C information-vector recovery
        #
        # After Patterson correction:
        #
        #     c_clean = a G.
        #
        # Since
        #
        #     G R = I_k,
        #
        #     a = c_clean R.
        #
        # R is verified during its construction. We deliberately
        # do not recompute G R during every decode operation.
        # --------------------------------------------------------

        if self.R is None:
            # Phase 2B serialized private keys do not yet contain R.
            # During the transition to the Phase 2C key format,
            # reconstruct it lazily once when necessary.
            (
                self.R,
                self.right_inverse_pivots,
            ) = self._construct_right_inverse()

        R = self._as_gf2_matrix(
            self.R
        )

        if R.arr.shape != (
            n,
            k,
        ):
            raise RuntimeError(
                f"Right inverse has shape "
                f"{R.arr.shape}; "
                f"expected ({n}, {k})."
            )

        # --------------------------------------------------------
        # Direct information extraction
        #
        # Cost after Patterson correction:
        #
        #     (1 x n)(n x k),
        #
        # approximately O(nk) binary operations.
        # --------------------------------------------------------

        message = (
            msg_arr
            * R
        )

        if message.arr.shape != (
            k,
        ):
            raise RuntimeError(
                "Right-inverse message recovery "
                f"produced shape {message.arr.shape}; "
                f"expected ({k},)."
            )

        return message

    # ============================================================
    # Decryption
    # ============================================================

    def decrypt(
        self,
        msg_arr,
    ):
        """
        Decrypt a McEliece ciphertext.

        Encryption has the form

            c_pub
              = m S G P + e_pub.

        Undoing the vector permutation gives

            c_sec
              = c_pub[P_inv]
              = m S G + e_sec.

        Patterson decoding produces the clean codeword

            (m S) G.

        Phase 2C then recovers

            m S
              = ((m S) G) R,

        because

            G R = I_k.

        Finally,

            m
              = (m S) S^{-1}.
        """

        if self.H is None:
            raise RuntimeError(
                "Private parity-check matrix H "
                "is unavailable."
            )

        if self.G is None:
            raise RuntimeError(
                "Private generator matrix G "
                "is unavailable."
            )

        if self.P_inv is None:
            raise RuntimeError(
                "Private inverse permutation "
                "P_inv is unavailable."
            )

        if self.S_inv is None:
            raise RuntimeError(
                "Private inverse scrambling "
                "matrix S_inv is unavailable."
            )

        H = self._as_gf2_matrix(
            self.H
        )

        S_inv = self._as_gf2_matrix(
            self.S_inv
        )

        ciphertext = (
            self._as_gf2_matrix(
                msg_arr
            )
        )

        n = H.arr.shape[1]

        if len(ciphertext) != n:
            raise ValueError(
                "Wrong ciphertext length. "
                f"Expected {n} bits, "
                f"received {len(ciphertext)}."
            )

        # --------------------------------------------------------
        # Normalize inverse permutation
        # --------------------------------------------------------

        P_inv = (
            self._as_permutation_vector(
                self.P_inv,
                n,
                name="P_inv",
            )
        )

        log.debug(
            f"ciphertext = {ciphertext}"
        )

        # --------------------------------------------------------
        # 1. Undo secret coordinate permutation
        #
        # If
        #
        #     y = x[P],
        #
        # then
        #
        #     x = y[P_inv].
        # --------------------------------------------------------

        secret_coordinate_word = (
            self._permute_vector(
                ciphertext,
                P_inv,
            )
        )

        log.debug(
            "after inverse permutation = "
            f"{secret_coordinate_word}"
        )

        # --------------------------------------------------------
        # 2. Patterson decode and right-inverse extraction
        #
        # The returned information vector is
        #
        #     m S.
        # --------------------------------------------------------

        scrambled_message = (
            self.decode(
                secret_coordinate_word
            )
        )

        log.debug(
            "decoded scrambled message = "
            f"{scrambled_message}"
        )

        # --------------------------------------------------------
        # 3. Undo scrambling
        #
        #     m = (m S) S^{-1}.
        # --------------------------------------------------------

        message = (
            scrambled_message
            * S_inv
        )

        if message.arr.shape != (
            self.G.arr.shape[0],
        ):
            raise RuntimeError(
                "Decryption produced unexpected "
                f"message shape {message.arr.shape}."
            )

        log.debug(
            f"plaintext = {message}"
        )

        return (
            message
            .to_numpy()
            .astype(
                np.uint8
            )
        )