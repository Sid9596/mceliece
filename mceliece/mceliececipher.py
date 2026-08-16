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

    Phase 2D representation
    -----------------------

    The implementation uses

        G       secret binary Goppa generator
        H       secret binary parity-check matrix
        S       secret scrambling matrix
        S_inv   inverse scrambling matrix
        P       secret permutation vector
        P_inv   inverse permutation vector
        Gp      public generator matrix

    together with a factorized generator right inverse.

    Let

        J = right_inverse_pivots

    contain k independent columns of G and let

        B = G[:, J].

    We store

        C = right_inverse_core = B^{-1}.

    Therefore

        B C = I_k.

    The conceptual full right inverse R would satisfy

        R[J, :] = C

    and all other rows would be zero, so

        G R = I_k.

    Phase 2D does not materialize R.

    For a clean secret codeword

        c_clean = a G,

    information extraction is performed directly as

        a
          = c_clean[J] C.

    The public generator is

        Gp = S G P,

    where P is represented by an integer permutation vector.
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

        # Secret Goppa code.
        self.G = None
        self.H = None
        self.k = None

        # --------------------------------------------------------
        # Phase 2D factorized right inverse
        #
        #     J = right_inverse_pivots
        #     C = right_inverse_core
        #
        # with
        #
        #     G[:, J] C = I_k.
        #
        # C has shape (k,k).
        # J has shape (k,).
        #
        # No dense n x k R is stored.
        # --------------------------------------------------------

        self.right_inverse_pivots = None
        self.right_inverse_core = None

        # McEliece permutation.
        self.P = None
        self.P_inv = None

        # Scrambling matrix.
        self.S = None
        self.S_inv = None

        # Public generator.
        self.Gp = None

        # Goppa polynomial data.
        self.g_poly = None
        self.irr_poly = None

    # ============================================================
    # Internal helpers
    # ============================================================

    @staticmethod
    def _as_gf2_matrix(value):
        """
        Normalize input as GF2Matrix.
        """

        if isinstance(
            value,
            GF2Matrix,
        ):
            return value

        arr = np.asarray(
            value
        )

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
        Validate and normalize a permutation vector.
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
            np.sort(
                arr
            ),
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
        Compare two GF2Matrix objects.
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
        Apply P to a row vector using

            x P := x[P].
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
        Apply a coordinate permutation to matrix columns.

            A P := A[:, P].
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
    # Phase 2D factorized right inverse
    # ============================================================

    def _construct_right_inverse_factorization(
        self,
    ):
        """
        Construct the factorized generator right inverse.

        Let

            G in GF(2)^{k x n}

        have rank k.

        Choose pivot-column indices

            J = (j_1,...,j_k)

        and define

            B = G[:,J].

        Compute

            C = B^{-1}.

        Then

            G[:,J] C = I_k.

        For any clean codeword

            c = aG,

        selecting coordinates J gives

            c[J]
              = a G[:,J]
              = aB.

        Hence

            c[J] C
              = a B B^{-1}
              = a.

        Returns
        -------
        pivots : ndarray, shape (k,)
            Selected independent column indices.

        core : GF2Matrix, shape (k,k)
            Inverse pivot submatrix B^{-1}.
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

        if k <= 0 or n <= 0:
            raise ValueError(
                "Generator dimensions must be positive."
            )

        if k > n:
            raise ValueError(
                "Generator matrix cannot have more "
                "rows than columns."
            )

        # --------------------------------------------------------
        # Compute RREF once and obtain pivot columns.
        # --------------------------------------------------------

        G_rref, rank = (
            G.rref()
        )

        if rank != k:
            raise RuntimeError(
                f"Generator matrix has rank {rank}; "
                f"expected {k}."
            )

        pivots = []

        for row_index in range(
            k
        ):
            pivot = None

            for column_index in range(
                n
            ):
                if int(
                    G_rref.arr[
                        row_index,
                        column_index,
                    ].n
                ) == 1:
                    pivot = column_index
                    break

            if pivot is None:
                raise RuntimeError(
                    "Unable to identify pivot "
                    f"for row {row_index}."
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
                "Pivot vector contains duplicates."
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
                "Pivot vector contains an "
                "out-of-range coordinate."
            )

        # --------------------------------------------------------
        # B = G[:,J].
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
                "Pivot submatrix has unexpected shape "
                f"{B.arr.shape}."
            )

        _, B_rank = (
            B.rref()
        )

        if B_rank != k:
            raise RuntimeError(
                "Pivot submatrix is singular."
            )

        # --------------------------------------------------------
        # C = B^{-1}.
        # --------------------------------------------------------

        core = (
            B.inv()
        )

        if core.arr.shape != (
            k,
            k,
        ):
            raise RuntimeError(
                "Right-inverse core has unexpected "
                f"shape {core.arr.shape}."
            )

        # --------------------------------------------------------
        # Verify B C = I_k.
        # --------------------------------------------------------

        identity = GF2Matrix.from_list(
            np.eye(
                k,
                dtype=np.uint8,
            )
        )

        BC = (
            B
            * core
        )

        if not self._gf2_matrix_equal(
            BC,
            identity,
        ):
            raise RuntimeError(
                "Right-inverse factorization failed: "
                "G[:,J] C != I_k."
            )

        # Since B is square and invertible, also verify C B.
        CB = (
            core
            * B
        )

        if not self._gf2_matrix_equal(
            CB,
            identity,
        ):
            raise RuntimeError(
                "Right-inverse factorization failed: "
                "C G[:,J] != I_k."
            )

        log.info(
            "Constructed factorized generator "
            "right inverse: "
            f"|J|={len(pivots)}, "
            f"C.shape={core.arr.shape}"
        )

        return (
            pivots,
            core,
        )

    def _validate_right_inverse_factorization(
        self,
    ):
        """
        Validate the currently stored factorized right inverse.

        Checks

            G[:,J] C = I_k.
        """

        if self.G is None:
            raise RuntimeError(
                "G is unavailable."
            )

        if self.right_inverse_pivots is None:
            raise RuntimeError(
                "right_inverse_pivots is unavailable."
            )

        if self.right_inverse_core is None:
            raise RuntimeError(
                "right_inverse_core is unavailable."
            )

        G = self._as_gf2_matrix(
            self.G
        )

        core = self._as_gf2_matrix(
            self.right_inverse_core
        )

        k, n = G.arr.shape

        pivots = np.asarray(
            self.right_inverse_pivots,
            dtype=np.int64,
        ).reshape(-1)

        if pivots.shape != (
            k,
        ):
            raise ValueError(
                "right_inverse_pivots has shape "
                f"{pivots.shape}; expected ({k},)."
            )

        if len(
            np.unique(
                pivots
            )
        ) != k:
            raise ValueError(
                "right_inverse_pivots contains "
                "duplicate coordinates."
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
            raise ValueError(
                "right_inverse_pivots contains "
                "an invalid coordinate."
            )

        if core.arr.shape != (
            k,
            k,
        ):
            raise ValueError(
                "right_inverse_core has shape "
                f"{core.arr.shape}; "
                f"expected ({k},{k})."
            )

        B = GF2Matrix(
            G.arr[
                :,
                pivots
            ]
        )

        identity = GF2Matrix.from_list(
            np.eye(
                k,
                dtype=np.uint8,
            )
        )

        if not self._gf2_matrix_equal(
            B * core,
            identity,
        ):
            raise ValueError(
                "Invalid right-inverse factorization: "
                "G[:,J] C != I_k."
            )

        return True

    def _extract_information_vector(
        self,
        clean_codeword,
    ):
        """
        Extract a from a clean codeword

            clean_codeword = a G

        using the Phase 2D factorization

            a = clean_codeword[J] C.

        No n x k right-inverse matrix is constructed.
        """

        clean_codeword = (
            self._as_gf2_matrix(
                clean_codeword
            )
        )

        if self.G is None:
            raise RuntimeError(
                "Generator matrix G is unavailable."
            )

        G = self._as_gf2_matrix(
            self.G
        )

        k, n = G.arr.shape

        if clean_codeword.arr.ndim != 1:
            raise ValueError(
                "Clean codeword must be one-dimensional."
            )

        if len(
            clean_codeword
        ) != n:
            raise ValueError(
                "Clean codeword has wrong length."
            )

        # Transitional lazy construction for in-memory objects.
        if (
            self.right_inverse_pivots is None
            or
            self.right_inverse_core is None
        ):
            (
                self.right_inverse_pivots,
                self.right_inverse_core,
            ) = (
                self
                ._construct_right_inverse_factorization()
            )

        pivots = np.asarray(
            self.right_inverse_pivots,
            dtype=np.int64,
        ).reshape(-1)

        core = self._as_gf2_matrix(
            self.right_inverse_core
        )

        if pivots.shape != (
            k,
        ):
            raise RuntimeError(
                "Unexpected pivot-vector dimensions."
            )

        if core.arr.shape != (
            k,
            k,
        ):
            raise RuntimeError(
                "Unexpected right-inverse-core "
                "dimensions."
            )

        # --------------------------------------------------------
        # Select only k coordinates:
        #
        #     clean_codeword[J].
        # --------------------------------------------------------

        selected = GF2Matrix(
            clean_codeword.arr[
                pivots
            ]
        )

        if selected.arr.shape != (
            k,
        ):
            raise RuntimeError(
                "Selected codeword coordinates "
                "have unexpected shape."
            )

        # --------------------------------------------------------
        # a = clean_codeword[J] C.
        # --------------------------------------------------------

        message = (
            selected
            * core
        )

        if message.arr.shape != (
            k,
        ):
            raise RuntimeError(
                "Information extraction produced "
                f"shape {message.arr.shape}; "
                f"expected ({k},)."
            )

        return message

    # ============================================================
    # Key generation
    # ============================================================

    def generate_random_keys(self):
        """
        Generate the binary-Goppa McEliece key pair.
        """

        # --------------------------------------------------------
        # 1. Goppa code.
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
        # 2. Serialize Goppa coefficients internally as bits.
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
        # 3. Actual dimension.
        # --------------------------------------------------------

        self.k = (
            self.G.arr.shape[0]
        )

        log.info(
            "Generated binary Goppa code "
            f"[n={self.n}, k={self.k}]"
        )

        # --------------------------------------------------------
        # 4. Phase 2D factorized right inverse.
        # --------------------------------------------------------

        (
            self.right_inverse_pivots,
            self.right_inverse_core,
        ) = (
            self
            ._construct_right_inverse_factorization()
        )

        self._validate_right_inverse_factorization()

        # --------------------------------------------------------
        # 5. Secret permutation.
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
                "Generated P is invalid."
            )

        if not np.array_equal(
            np.sort(
                self.P_inv
            ),
            identity_positions,
        ):
            raise RuntimeError(
                "Generated P_inv is invalid."
            )

        if not np.array_equal(
            self.P[
                self.P_inv
            ],
            identity_positions,
        ):
            raise RuntimeError(
                "P[P_inv] != identity."
            )

        if not np.array_equal(
            self.P_inv[
                self.P
            ],
            identity_positions,
        ):
            raise RuntimeError(
                "P_inv[P] != identity."
            )

        # --------------------------------------------------------
        # 6. Scrambling matrix.
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
        # 7. Public generator.
        #
        #     G_pub = S G P.
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

        if self.Gp.arr.shape != (
            self.k,
            self.n,
        ):
            raise RuntimeError(
                "Public generator has unexpected "
                f"shape {self.Gp.arr.shape}."
            )

        log.info(
            "McEliece Phase 2D key generation completed."
        )

    # ============================================================
    # Encryption
    # ============================================================

    def encrypt(
        self,
        msg_arr,
    ):
        """
        Baseline McEliece encryption

            c = m G_pub + e,

        with

            wt(e) = t.
        """

        if self.Gp is None:
            raise RuntimeError(
                "Public key is unavailable."
            )

        Gp = self._as_gf2_matrix(
            self.Gp
        )

        k = Gp.arr.shape[0]
        n = Gp.arr.shape[1]

        msg_arr = np.asarray(
            msg_arr,
            dtype=np.uint8,
        ).reshape(-1)

        if msg_arr.shape != (
            k,
        ):
            raise ValueError(
                "Wrong plaintext length. "
                f"Expected {k}, "
                f"received {len(msg_arr)}."
            )

        if not np.all(
            (msg_arr == 0)
            |
            (msg_arr == 1)
        ):
            raise ValueError(
                "Plaintext must be binary."
            )

        if self.t > n:
            raise ValueError(
                "Error weight exceeds code length."
            )

        message = GF2Matrix.from_list(
            msg_arr
        )

        ciphertext = (
            message
            * Gp
        )

        bits_to_flip = np.random.choice(
            n,
            size=self.t,
            replace=False,
        )

        for position in (
            bits_to_flip
        ):
            ciphertext[
                position
            ] = ciphertext[
                position
            ].flip()

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
        Patterson-style error correction.

        Current support:

            1, alpha, alpha^2, ...
        """

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
            len(
                syndrome_flat
            )
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

        S_inv_poly = (
            S_poly.inv_mod(
                self.g_poly
            )
        )

        # --------------------------------------------------------
        # Special low-degree case.
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
            g0, g1 = (
                self.g_poly.split()
            )

            g1_inv = (
                g1.inv_mod(
                    self.g_poly
                )
            )

            w = (
                g0
                * g1_inv
            )

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

            H0, H1 = (
                H_poly.split()
            )

            # Patterson auxiliary polynomial.
            R_poly = (
                H0
                +
                w * H1
            )

            b, _, a = (
                ext_euclid_poly_alt(
                    R_poly,
                    self.g_poly,
                    ring,
                    self.t,
                )
            )

            tau_poly = (
                a ** 2
                +
                GF2mPoly.x(
                    ring
                )
                * b ** 2
            )

        # --------------------------------------------------------
        # Locate and repair errors.
        # --------------------------------------------------------

        test_elem = (
            ring.one()
        )

        for i in range(
            len(
                msg_arr
            )
        ):
            value = (
                tau_poly.eval(
                    test_elem
                )
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

        1. Compute syndrome.
        2. Apply Patterson correction when necessary.
        3. Extract information using

               a = c_clean[J] C.

        The full n x k right inverse is never constructed.
        """

        if self.G is None:
            raise RuntimeError(
                "Secret generator G is unavailable."
            )

        if self.H is None:
            raise RuntimeError(
                "Parity-check matrix H is unavailable."
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

        if len(
            msg_arr
        ) != n:
            raise ValueError(
                "Received-word length does not "
                "match code length."
            )

        # Syndrome.
        syndrome = (
            msg_arr
            * H.T()
        )

        syndrome_is_zero = all(
            int(x.n) == 0
            for x
            in syndrome.arr.flat
        )

        # Patterson correction.
        if not syndrome_is_zero:
            msg_arr = (
                self.repair_errors(
                    msg_arr,
                    syndrome,
                )
            )

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

        # Phase 2D factorized extraction.
        return (
            self._extract_information_vector(
                msg_arr
            )
        )

    # ============================================================
    # Decryption
    # ============================================================

    def decrypt(
        self,
        msg_arr,
    ):
        """
        Decrypt

            c_pub = m S G P + e.

        The pipeline is

            c_pub
                ->
            c_pub[P_inv]
                ->
            Patterson
                ->
            (mS)G
                ->
            select J
                ->
            ((mS)G)[J] C
                ->
            mS
                ->
            (mS)S_inv
                ->
            m.
        """

        if self.H is None:
            raise RuntimeError(
                "Private H is unavailable."
            )

        if self.G is None:
            raise RuntimeError(
                "Private G is unavailable."
            )

        if self.P_inv is None:
            raise RuntimeError(
                "Private P_inv is unavailable."
            )

        if self.S_inv is None:
            raise RuntimeError(
                "Private S_inv is unavailable."
            )

        if (
            self.right_inverse_pivots is None
            or
            self.right_inverse_core is None
        ):
            raise RuntimeError(
                "Factorized right inverse is unavailable."
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

        if len(
            ciphertext
        ) != n:
            raise ValueError(
                "Wrong ciphertext length. "
                f"Expected {n}, "
                f"received {len(ciphertext)}."
            )

        P_inv = (
            self._as_permutation_vector(
                self.P_inv,
                n,
                name="P_inv",
            )
        )

        # Undo public coordinate permutation.
        secret_coordinate_word = (
            self._permute_vector(
                ciphertext,
                P_inv,
            )
        )

        # Patterson + factorized information extraction.
        scrambled_message = (
            self.decode(
                secret_coordinate_word
            )
        )

        # Undo S.
        message = (
            scrambled_message
            * S_inv
        )

        k = self.G.arr.shape[0]

        if message.arr.shape != (
            k,
        ):
            raise RuntimeError(
                "Decryption produced unexpected "
                f"shape {message.arr.shape}."
            )

        return (
            message
            .to_numpy()
            .astype(
                np.uint8
            )
        )