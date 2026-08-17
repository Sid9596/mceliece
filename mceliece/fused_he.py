"""
Fused evaluation-key layer for the binary-Goppa McEliece backend.

Phase 3A
--------
Construct a randomized secret right inverse

    R_HE = R_0 + H^T X

with

    G R_HE = I_k,

and transport it to public coordinates:

    Lambda = P^{-1} R_HE S^{-1}.

Under the repository's permutation-array convention

    x P = x[P],

this is represented as

    Lambda = (R_HE S_inv)[P, :].

The implementation verifies

    G_pub Lambda = I_k.


Phase 3B
--------
The plaintext algebra is truncated polynomial multiplication

    a odot b
        =
    (a(X)b(X)) mod X^k.

For each public-coordinate pair

    0 <= u <= v < n,

construct

    K_{u,v}
        =
    (Lambda_u odot Lambda_v) G_pub.


Phase 3C
--------
Add a low-weight fused perturbation

    z_{u,v},

with

    wt(z_{u,v}) = tau_F,

and publish

    Y_{u,v}
        =
    K_{u,v} + z_{u,v}.

Because odot is commutative, only u <= v is stored:

    N_FEK = n(n+1)/2.

Rows are bit-packed with numpy.packbits.


Phase 3D
--------
For ciphertexts c1,c2, compressed symmetric evaluation uses

    diagonal:
        s_{u,u} = c1_u c2_u

    off-diagonal:
        s_{u,v}
          =
        c1_u c2_v + c1_v c2_u.

Then

    EvalMult(c1,c2)
        =
    sum_{u <= v} s_{u,v} Y_{u,v}.


HE encryption
-------------
A plaintext mu in GF(2)^ell is lifted to

    a = (mu || tail) in GF(2)^k.

Encryption uses

    c = a G_pub + e,

where the secret admissibility condition is

    e Lambda_msg = 0,

and Lambda_msg consists of the first ell columns of Lambda.

This implementation uses rejection sampling for low-weight admissible
errors.  That is suitable for experiments and small ell.  It is not
a scalable public admissible-noise sampler.

The baseline McEliece encrypt() method is intentionally not modified.
"""

from pathlib import Path
import time

import numpy as np

from mceliece.mathutils import GF2Matrix


FUSED_EVK_FORMAT_VERSION = 1


# ======================================================================
# GF(2) helpers
# ======================================================================

def gf2_uint8(value):
    """
    Convert GF2Matrix / NumPy-like input to a binary uint8 array
    while preserving its shape.
    """

    if isinstance(
        value,
        GF2Matrix,
    ):
        arr = value.arr

    else:
        arr = np.asarray(
            value
        )

        if arr.ndim == 0:
            item = arr.item()

            if isinstance(
                item,
                GF2Matrix,
            ):
                arr = item.arr

    flat = np.fromiter(
        (
            int(entry.n)
            if hasattr(
                entry,
                "n",
            )
            else int(entry)
            for entry in arr.flat
        ),
        dtype=np.uint8,
        count=arr.size,
    )

    result = flat.reshape(
        arr.shape
    )

    result &= np.uint8(
        1
    )

    return result


def gf2_vector(
    value,
    expected_length=None,
    name="vector",
):
    """
    Normalize input to a one-dimensional binary uint8 vector.
    """

    result = (
        gf2_uint8(
            value
        )
        .reshape(-1)
    )

    if (
        expected_length is not None
        and
        result.shape
        !=
        (
            int(
                expected_length
            ),
        )
    ):
        raise ValueError(
            f"{name} has shape "
            f"{result.shape}; expected "
            f"({expected_length},)."
        )

    return result


def gf2_matmul(
    A,
    B,
):
    """
    Matrix multiplication over GF(2).

    For the parameter ranges considered here the shared dimension is
    well below 2^16, so uint16 accumulation is exact.

    A uint32 fallback is retained for larger shared dimensions.
    """

    A = np.asarray(
        A,
        dtype=np.uint8,
    )

    B = np.asarray(
        B,
        dtype=np.uint8,
    )

    if A.ndim == 0:
        raise ValueError(
            "A must have at least one dimension."
        )

    if B.ndim == 0:
        raise ValueError(
            "B must have at least one dimension."
        )

    shared = int(
        A.shape[-1]
    )

    if shared != int(
        B.shape[0]
    ):
        raise ValueError(
            "GF(2) matrix dimensions do not agree: "
            f"{A.shape} @ {B.shape}."
        )

    if shared <= 65535:
        accumulator_dtype = (
            np.uint16
        )

    else:
        accumulator_dtype = (
            np.uint32
        )

    result = (
        A.astype(
            accumulator_dtype,
            copy=False,
        )
        @
        B.astype(
            accumulator_dtype,
            copy=False,
        )
    )

    return (
        result
        &
        1
    ).astype(
        np.uint8
    )


def gf2_identity(
    k,
):
    return np.eye(
        int(k),
        dtype=np.uint8,
    )


def gf2_equal(
    A,
    B,
):
    return np.array_equal(
        np.asarray(
            A,
            dtype=np.uint8,
        ),
        np.asarray(
            B,
            dtype=np.uint8,
        ),
    )


def hamming_weight(
    vector,
):
    return int(
        np.count_nonzero(
            np.asarray(
                vector,
                dtype=np.uint8,
            )
        )
    )


# ======================================================================
# Plaintext algebra
# ======================================================================

def truncated_product(
    a,
    b,
    output_length=None,
):
    """
    Truncated polynomial multiplication over GF(2).

    Interpret

        a = (a_0,...,a_{r-1})
        b = (b_0,...,b_{s-1})

    as

        a(X) = sum a_i X^i,
        b(X) = sum b_i X^i.

    Return coefficients of

        a(X)b(X) mod X^L,

    where L = output_length.

    If output_length is omitted, L = min(len(a),len(b)).
    """

    a = gf2_vector(
        a,
        name="a",
    )

    b = gf2_vector(
        b,
        name="b",
    )

    if output_length is None:
        output_length = min(
            len(a),
            len(b),
        )

    output_length = int(
        output_length
    )

    if output_length <= 0:
        raise ValueError(
            "output_length must be positive."
        )

    shared_bound = max(
        len(a),
        len(b),
    )

    dtype = (
        np.uint16
        if shared_bound <= 65535
        else np.uint32
    )

    convolution = np.convolve(
        a.astype(
            dtype,
            copy=False,
        ),
        b.astype(
            dtype,
            copy=False,
        ),
    )

    result = np.zeros(
        output_length,
        dtype=np.uint8,
    )

    usable = min(
        output_length,
        len(
            convolution
        ),
    )

    result[
        :usable
    ] = (
        convolution[
            :usable
        ]
        &
        1
    ).astype(
        np.uint8
    )

    return result


# ======================================================================
# Symmetric-pair indexing
# ======================================================================

def fused_pair_count(
    n,
):
    n = int(
        n
    )

    return (
        n
        *
        (
            n + 1
        )
        //
        2
    )


def fused_pair_index(
    u,
    v,
    n,
):
    """
    Row index for pairs ordered as

        (0,0),(0,1),...,(0,n-1),
        (1,1),(1,2),...,
        ...
        (n-1,n-1).
    """

    n = int(
        n
    )

    u = int(
        u
    )

    v = int(
        v
    )

    if not (
        0
        <=
        u
        <=
        v
        <
        n
    ):
        raise ValueError(
            "Pair must satisfy "
            "0 <= u <= v < n."
        )

    prefix = (
        u * n
        -
        (
            u
            *
            (
                u - 1
            )
            //
            2
        )
    )

    return (
        prefix
        +
        (
            v - u
        )
    )


# ======================================================================
# Public fused evaluation key
# ======================================================================

class FusedEvaluationKey:
    """
    Public compressed symmetric table

        Y_{u,v},  0 <= u <= v < n.

    Each n-bit row is bit-packed.
    """

    def __init__(
        self,
        n,
        k,
        ell,
        tau_F,
        Y_packed,
        generation_seconds=0.0,
    ):
        self.n = int(
            n
        )

        self.k = int(
            k
        )

        self.ell = int(
            ell
        )

        self.tau_F = int(
            tau_F
        )

        self.pair_count = (
            fused_pair_count(
                self.n
            )
        )

        self.row_bytes = (
            (
                self.n
                +
                7
            )
            //
            8
        )

        self.Y_packed = np.asarray(
            Y_packed,
            dtype=np.uint8,
        )

        expected_shape = (
            self.pair_count,
            self.row_bytes,
        )

        if (
            self.Y_packed.shape
            !=
            expected_shape
        ):
            raise ValueError(
                "Y_packed has shape "
                f"{self.Y_packed.shape}; "
                f"expected {expected_shape}."
            )

        self.generation_seconds = float(
            generation_seconds
        )

    @property
    def packed_bytes(
        self,
    ):
        return int(
            self.Y_packed.nbytes
        )

    @property
    def unpacked_uint8_bytes(
        self,
    ):
        return int(
            self.pair_count
            *
            self.n
        )

    def get_row(
        self,
        u,
        v,
    ):
        """
        Return Y_{u,v} as an n-bit uint8 vector.
        """

        if u > v:
            u, v = (
                v,
                u,
            )

        index = fused_pair_index(
            u,
            v,
            self.n,
        )

        packed = (
            self.Y_packed[
                index
            ]
        )

        return (
            np.unpackbits(
                packed,
                bitorder="little",
            )[
                :self.n
            ]
            .astype(
                np.uint8
            )
        )

    def evaluate_multiply(
        self,
        ciphertext_left,
        ciphertext_right,
        return_selector_count=False,
    ):
        """
        Evaluate the compressed symmetric fused table.

        For u < v the selector is

            c1_u c2_v + c1_v c2_u.

        For u = v the selector is

            c1_u c2_u.
        """

        c1 = gf2_vector(
            ciphertext_left,
            expected_length=self.n,
            name="ciphertext_left",
        )

        c2 = gf2_vector(
            ciphertext_right,
            expected_length=self.n,
            name="ciphertext_right",
        )

        accumulator = np.zeros(
            self.row_bytes,
            dtype=np.uint8,
        )

        selected_count = 0

        offset = 0

        for u in range(
            self.n
        ):
            block_length = (
                self.n
                -
                u
            )

            selectors = np.empty(
                block_length,
                dtype=np.uint8,
            )

            # Diagonal pair (u,u).
            selectors[0] = (
                c1[u]
                &
                c2[u]
            )

            # Pairs (u,v), v > u.
            if block_length > 1:
                selectors[
                    1:
                ] = (
                    (
                        c1[u]
                        &
                        c2[
                            u + 1:
                        ]
                    )
                    ^
                    (
                        c2[u]
                        &
                        c1[
                            u + 1:
                        ]
                    )
                )

            relative_indices = (
                np.flatnonzero(
                    selectors
                )
            )

            if (
                relative_indices.size
                >
                0
            ):
                rows = (
                    self.Y_packed[
                        offset
                        +
                        relative_indices
                    ]
                )

                block_xor = (
                    np.bitwise_xor.reduce(
                        rows,
                        axis=0,
                    )
                )

                accumulator ^= (
                    block_xor
                )

                selected_count += int(
                    relative_indices.size
                )

            offset += (
                block_length
            )

        if offset != self.pair_count:
            raise RuntimeError(
                "Internal fused-table indexing error."
            )

        result = (
            np.unpackbits(
                accumulator,
                bitorder="little",
            )[
                :self.n
            ]
            .astype(
                np.uint8
            )
        )

        result_gf2 = (
            GF2Matrix.from_list(
                result
            )
        )

        if return_selector_count:
            return (
                result_gf2,
                selected_count,
            )

        return result_gf2

    def save(
        self,
        filename,
    ):
        """
        Serialize the public fused evaluation key.
        """

        filename = Path(
            filename
        )

        np.savez_compressed(
            filename,

            format_version=np.asarray(
                FUSED_EVK_FORMAT_VERSION,
                dtype=np.int64,
            ),

            n=np.asarray(
                self.n,
                dtype=np.int64,
            ),

            k=np.asarray(
                self.k,
                dtype=np.int64,
            ),

            ell=np.asarray(
                self.ell,
                dtype=np.int64,
            ),

            tau_F=np.asarray(
                self.tau_F,
                dtype=np.int64,
            ),

            generation_seconds=np.asarray(
                self.generation_seconds,
                dtype=np.float64,
            ),

            Y_packed=self.Y_packed,
        )

    @classmethod
    def load(
        cls,
        filename,
    ):
        """
        Load a serialized public fused evaluation key.
        """

        with np.load(
            filename,
            allow_pickle=False,
        ) as data:

            version = int(
                np.asarray(
                    data[
                        "format_version"
                    ]
                ).item()
            )

            if (
                version
                !=
                FUSED_EVK_FORMAT_VERSION
            ):
                raise ValueError(
                    "Unsupported fused-evaluation-key "
                    f"format {version}."
                )

            n = int(
                np.asarray(
                    data[
                        "n"
                    ]
                ).item()
            )

            k = int(
                np.asarray(
                    data[
                        "k"
                    ]
                ).item()
            )

            ell = int(
                np.asarray(
                    data[
                        "ell"
                    ]
                ).item()
            )

            tau_F = int(
                np.asarray(
                    data[
                        "tau_F"
                    ]
                ).item()
            )

            generation_seconds = float(
                np.asarray(
                    data[
                        "generation_seconds"
                    ]
                ).item()
            )

            Y_packed = np.asarray(
                data[
                    "Y_packed"
                ],
                dtype=np.uint8,
            ).copy()

        return cls(
            n=n,
            k=k,
            ell=ell,
            tau_F=tau_F,
            Y_packed=Y_packed,
            generation_seconds=(
                generation_seconds
            ),
        )


# ======================================================================
# Secret HE context
# ======================================================================

class FusedMcElieceHE:
    """
    Secret-key HE context layered over an existing McElieceCipher.

    The underlying McEliece implementation remains unchanged.
    """

    def __init__(
        self,
        mc,
        ell,
        tau_enc=0,
        tau_F=0,
        seed=None,
    ):
        self.mc = mc

        self.ell = int(
            ell
        )

        self.tau_enc = int(
            tau_enc
        )

        self.tau_F = int(
            tau_F
        )

        self.rng = (
            np.random.default_rng(
                seed
            )
        )

        self._validate_backend()

        self.n = int(
            self.mc.n
        )

        self.k = int(
            self.mc.k
        )

        self.decoder_radius = int(
            self.mc.t
        )

        if not (
            1
            <=
            self.ell
            <=
            self.k
        ):
            raise ValueError(
                "ell must satisfy "
                "1 <= ell <= k."
            )

        if not (
            0
            <=
            self.tau_enc
            <=
            self.decoder_radius
        ):
            raise ValueError(
                "tau_enc must satisfy "
                "0 <= tau_enc <= t."
            )

        if not (
            0
            <=
            self.tau_F
            <=
            self.n
        ):
            raise ValueError(
                "tau_F must satisfy "
                "0 <= tau_F <= n."
            )

        self.Lambda = None

        # Optional diagnostic storage.
        self.he_right_inverse = None

        self.transport_diagnostics = {}

        self.evk = None

    # ==================================================================
    # Backend validation
    # ==================================================================

    def _validate_backend(
        self,
    ):
        required = (
            "G",
            "H",
            "S_inv",
            "P",
            "Gp",
            "right_inverse_pivots",
            "right_inverse_core",
            "k",
            "n",
            "t",
        )

        missing = [
            name
            for name in required
            if getattr(
                self.mc,
                name,
                None,
            ) is None
        ]

        if missing:
            raise RuntimeError(
                "McEliece private backend is incomplete. "
                f"Missing: {missing}"
            )

    # ==================================================================
    # Phase 3A: randomized HE right inverse and Lambda
    # ==================================================================

    def construct_transport(
        self,
        randomization_trials=32,
        store_right_inverse=False,
    ):
        """
        Construct

            R_HE = R_0 + H^T X

        and

            Lambda = P^{-1} R_HE S^{-1}.

        Under the current permutation-array convention,

            Lambda = (R_HE S_inv)[P,:].

        The implementation verifies

            G R_HE = I_k

        and

            G_pub Lambda = I_k.
        """

        randomization_trials = int(
            randomization_trials
        )

        if randomization_trials <= 0:
            raise ValueError(
                "randomization_trials must be positive."
            )

        G = gf2_uint8(
            self.mc.G
        )

        H = gf2_uint8(
            self.mc.H
        )

        S_inv = gf2_uint8(
            self.mc.S_inv
        )

        G_pub = gf2_uint8(
            self.mc.Gp
        )

        P = np.asarray(
            self.mc.P,
            dtype=np.int64,
        ).reshape(-1)

        J = np.asarray(
            self.mc.right_inverse_pivots,
            dtype=np.int64,
        ).reshape(-1)

        C = gf2_uint8(
            self.mc.right_inverse_core
        )

        if G.shape != (
            self.k,
            self.n,
        ):
            raise RuntimeError(
                "Unexpected G dimensions."
            )

        if H.ndim != 2:
            raise RuntimeError(
                "Unexpected H dimensions."
            )

        if H.shape[1] != self.n:
            raise RuntimeError(
                "H has wrong code length."
            )

        if S_inv.shape != (
            self.k,
            self.k,
        ):
            raise RuntimeError(
                "S_inv has wrong dimensions."
            )

        if G_pub.shape != (
            self.k,
            self.n,
        ):
            raise RuntimeError(
                "G_pub has wrong dimensions."
            )

        if J.shape != (
            self.k,
        ):
            raise RuntimeError(
                "Pivot vector has wrong dimensions."
            )

        if C.shape != (
            self.k,
            self.k,
        ):
            raise RuntimeError(
                "Right-inverse core has wrong dimensions."
            )

        # --------------------------------------------------------------
        # Sparse Phase-2D conceptual right inverse.
        #
        # This is used only as one point in the affine right-inverse
        # space.  It is not used directly as the HE transport inverse.
        # --------------------------------------------------------------

        R0 = np.zeros(
            (
                self.n,
                self.k,
            ),
            dtype=np.uint8,
        )

        R0[
            J,
            :
        ] = C

        identity = (
            gf2_identity(
                self.k
            )
        )

        if not gf2_equal(
            gf2_matmul(
                G,
                R0,
            ),
            identity,
        ):
            raise RuntimeError(
                "Phase-2D conceptual right inverse "
                "does not satisfy G R0 = I_k."
            )

        base_zero_rows = int(
            np.sum(
                np.all(
                    R0 == 0,
                    axis=1,
                )
            )
        )

        # --------------------------------------------------------------
        # Randomize within the affine right-inverse space:
        #
        #     R = R0 + H^T X.
        #
        # Since GH^T = 0:
        #
        #     GR = I.
        #
        # Among sampled candidates choose the one with the fewest
        # zero rows.
        # --------------------------------------------------------------

        parity_rows = int(
            H.shape[0]
        )

        best_R = None
        best_zero_rows = (
            self.n + 1
        )

        successful_candidates = 0

        start = (
            time.perf_counter()
        )

        for _ in range(
            randomization_trials
        ):
            X = self.rng.integers(
                0,
                2,
                size=(
                    parity_rows,
                    self.k,
                ),
                dtype=np.uint8,
            )

            kernel_term = (
                gf2_matmul(
                    H.T,
                    X,
                )
            )

            candidate = (
                R0
                ^
                kernel_term
            )

            GR = (
                gf2_matmul(
                    G,
                    candidate,
                )
            )

            if not gf2_equal(
                GR,
                identity,
            ):
                continue

            successful_candidates += 1

            zero_rows = int(
                np.sum(
                    np.all(
                        candidate == 0,
                        axis=1,
                    )
                )
            )

            if (
                zero_rows
                <
                best_zero_rows
            ):
                best_zero_rows = (
                    zero_rows
                )

                best_R = (
                    candidate.copy()
                )

            if zero_rows == 0:
                break

        randomization_seconds = (
            time.perf_counter()
            -
            start
        )

        if best_R is None:
            raise RuntimeError(
                "Unable to construct a randomized "
                "HE right inverse."
            )

        if not gf2_equal(
            gf2_matmul(
                G,
                best_R,
            ),
            identity,
        ):
            raise RuntimeError(
                "Randomized HE right inverse failed "
                "G R_HE = I_k."
            )

        # --------------------------------------------------------------
        # First multiply on the right by S^{-1}.
        # --------------------------------------------------------------

        RSinv = (
            gf2_matmul(
                best_R,
                S_inv,
            )
        )

        # --------------------------------------------------------------
        # Apply P^{-1} on the left.
        #
        # For the repository convention
        #
        #     xP = x[P],
        #
        # the array representation of P^{-1} M is
        #
        #     M[P,:].
        # --------------------------------------------------------------

        Lambda = (
            RSinv[
                P,
                :
            ]
            .copy()
        )

        if Lambda.shape != (
            self.n,
            self.k,
        ):
            raise RuntimeError(
                "Lambda has wrong dimensions."
            )

        public_identity = (
            gf2_matmul(
                G_pub,
                Lambda,
            )
        )

        if not gf2_equal(
            public_identity,
            identity,
        ):
            raise RuntimeError(
                "Transport construction failed: "
                "G_pub Lambda != I_k. "
                "This usually indicates a permutation-"
                "orientation error."
            )

        lambda_zero_rows = int(
            np.sum(
                np.all(
                    Lambda == 0,
                    axis=1,
                )
            )
        )

        row_weights = np.sum(
            Lambda,
            axis=1,
            dtype=np.int64,
        )

        self.Lambda = (
            Lambda
        )

        if store_right_inverse:
            self.he_right_inverse = (
                best_R.copy()
            )

        else:
            self.he_right_inverse = None

        self.transport_diagnostics = {
            "base_zero_rows": (
                base_zero_rows
            ),
            "randomized_zero_rows": (
                best_zero_rows
            ),
            "lambda_zero_rows": (
                lambda_zero_rows
            ),
            "successful_candidates": (
                successful_candidates
            ),
            "randomization_trials_requested": (
                randomization_trials
            ),
            "randomization_seconds": (
                randomization_seconds
            ),
            "lambda_row_weight_min": int(
                np.min(
                    row_weights
                )
            ),
            "lambda_row_weight_max": int(
                np.max(
                    row_weights
                )
            ),
            "lambda_row_weight_mean": float(
                np.mean(
                    row_weights
                )
            ),
            "GpLambda_identity": True,
        }

        return (
            self.Lambda
        )

    # ==================================================================
    # Lambda helpers
    # ==================================================================

    def _require_lambda(
        self,
    ):
        if self.Lambda is None:
            raise RuntimeError(
                "Lambda has not been constructed. "
                "Call construct_transport() first."
            )

    @property
    def Lambda_msg(
        self,
    ):
        self._require_lambda()

        return (
            self.Lambda[
                :,
                :self.ell
            ]
        )

    def project_ciphertext(
        self,
        ciphertext,
    ):
        """
        Compute

            ciphertext * Lambda.
        """

        self._require_lambda()

        ciphertext = gf2_vector(
            ciphertext,
            expected_length=self.n,
            name="ciphertext",
        )

        return (
            gf2_matmul(
                ciphertext,
                self.Lambda,
            )
        )

    # ==================================================================
    # Admissible error sampling
    # ==================================================================

    def sample_admissible_error(
        self,
        weight=None,
        max_attempts=100000,
    ):
        """
        Sample a weight-w error satisfying

            e Lambda_msg = 0.

        This is a secret rejection sampler intended for experiments.

        It does not solve the public admissible-noise-sampling problem.
        """

        self._require_lambda()

        if weight is None:
            weight = (
                self.tau_enc
            )

        weight = int(
            weight
        )

        max_attempts = int(
            max_attempts
        )

        if not (
            0
            <=
            weight
            <=
            self.decoder_radius
        ):
            raise ValueError(
                "Admissible encryption-error weight "
                "must be between 0 and the "
                "Goppa decoding radius t."
            )

        if weight > self.n:
            raise ValueError(
                "Error weight exceeds code length."
            )

        if weight == 0:
            return np.zeros(
                self.n,
                dtype=np.uint8,
            )

        Lambda_msg = (
            self.Lambda_msg
        )

        # --------------------------------------------------------------
        # Fast exact handling for weight one.
        # --------------------------------------------------------------

        if weight == 1:
            candidates = (
                np.flatnonzero(
                    np.all(
                        Lambda_msg
                        ==
                        0,
                        axis=1,
                    )
                )
            )

            if candidates.size == 0:
                raise RuntimeError(
                    "No weight-1 admissible error exists "
                    "for the current Lambda_msg."
                )

            position = int(
                self.rng.choice(
                    candidates
                )
            )

            error = np.zeros(
                self.n,
                dtype=np.uint8,
            )

            error[
                position
            ] = 1

            return error

        # --------------------------------------------------------------
        # General bounded-weight rejection sampling.
        # --------------------------------------------------------------

        for _ in range(
            max_attempts
        ):
            support = (
                self.rng.choice(
                    self.n,
                    size=weight,
                    replace=False,
                )
            )

            projected_error = (
                np.bitwise_xor.reduce(
                    Lambda_msg[
                        support
                    ],
                    axis=0,
                )
            )

            if not np.any(
                projected_error
            ):
                error = np.zeros(
                    self.n,
                    dtype=np.uint8,
                )

                error[
                    support
                ] = 1

                return error

        raise RuntimeError(
            "Unable to sample an admissible "
            f"weight-{weight} error after "
            f"{max_attempts} attempts. "
            "This rejection sampler is intended "
            "for small experimental parameters."
        )

    # ==================================================================
    # HE encryption / decryption
    # ==================================================================

    def encrypt(
        self,
        plaintext,
        return_metadata=False,
    ):
        """
        Encrypt mu in GF(2)^ell.

        A fresh random tail gives

            a = (mu || tail) in GF(2)^k.

        Then

            c = a G_pub + e,

        where

            e Lambda_msg = 0.
        """

        self._require_lambda()

        mu = gf2_vector(
            plaintext,
            expected_length=self.ell,
            name="plaintext",
        )

        tail_length = (
            self.k
            -
            self.ell
        )

        tail = (
            self.rng.integers(
                0,
                2,
                size=tail_length,
                dtype=np.uint8,
            )
        )

        a = np.concatenate(
            (
                mu,
                tail,
            )
        ).astype(
            np.uint8
        )

        G_pub = (
            gf2_uint8(
                self.mc.Gp
            )
        )

        clean = (
            gf2_matmul(
                a,
                G_pub,
            )
        )

        error = (
            self.sample_admissible_error(
                self.tau_enc
            )
        )

        # Defensive verification.
        projected_error = (
            gf2_matmul(
                error,
                self.Lambda_msg,
            )
        )

        if np.any(
            projected_error
        ):
            raise RuntimeError(
                "Internal admissible-error sampler "
                "produced e Lambda_msg != 0."
            )

        ciphertext = (
            clean
            ^
            error
        )

        ciphertext_gf2 = (
            GF2Matrix.from_list(
                ciphertext
            )
        )

        if return_metadata:
            return (
                ciphertext_gf2,
                {
                    "mu": (
                        mu.copy()
                    ),
                    "tail": (
                        tail.copy()
                    ),
                    "a": (
                        a.copy()
                    ),
                    "clean": (
                        clean.copy()
                    ),
                    "error": (
                        error.copy()
                    ),
                    "error_weight": (
                        hamming_weight(
                            error
                        )
                    ),
                },
            )

        return (
            ciphertext_gf2
        )

    def decrypt_full(
        self,
        ciphertext,
    ):
        """
        Decode the complete k-bit lifted plaintext representative.
        """

        result = (
            self.mc.decrypt(
                ciphertext
            )
        )

        result = gf2_vector(
            result,
            expected_length=self.k,
            name="decrypted representative",
        )

        return result

    def decrypt(
        self,
        ciphertext,
    ):
        """
        Return the message coordinates only.
        """

        representative = (
            self.decrypt_full(
                ciphertext
            )
        )

        return (
            representative[
                :self.ell
            ]
            .copy()
        )

    # ==================================================================
    # Plaintext reference operations
    # ==================================================================

    def plaintext_add(
        self,
        left,
        right,
    ):
        left = gf2_vector(
            left,
            expected_length=self.ell,
            name="left plaintext",
        )

        right = gf2_vector(
            right,
            expected_length=self.ell,
            name="right plaintext",
        )

        return (
            left
            ^
            right
        )

    def plaintext_multiply(
        self,
        left,
        right,
    ):
        left = gf2_vector(
            left,
            expected_length=self.ell,
            name="left plaintext",
        )

        right = gf2_vector(
            right,
            expected_length=self.ell,
            name="right plaintext",
        )

        return (
            truncated_product(
                left,
                right,
                output_length=self.ell,
            )
        )

    # ==================================================================
    # Ciphertext addition
    # ==================================================================

    def eval_add(
        self,
        ciphertext_left,
        ciphertext_right,
    ):
        left = gf2_vector(
            ciphertext_left,
            expected_length=self.n,
            name="ciphertext_left",
        )

        right = gf2_vector(
            ciphertext_right,
            expected_length=self.n,
            name="ciphertext_right",
        )

        return (
            GF2Matrix.from_list(
                left
                ^
                right
            )
        )

    # ==================================================================
    # Phase 3B/3C: K_{u,v} and Y_{u,v}
    # ==================================================================

    def clean_fused_row(
        self,
        u,
        v,
    ):
        """
        Compute

            K_{u,v}
                =
            (Lambda_u odot Lambda_v) G_pub.
        """

        self._require_lambda()

        u = int(
            u
        )

        v = int(
            v
        )

        if not (
            0
            <=
            u
            <
            self.n
        ):
            raise ValueError(
                "u is outside the ciphertext range."
            )

        if not (
            0
            <=
            v
            <
            self.n
        ):
            raise ValueError(
                "v is outside the ciphertext range."
            )

        product = (
            truncated_product(
                self.Lambda[
                    u
                ],
                self.Lambda[
                    v
                ],
                output_length=self.k,
            )
        )

        G_pub = (
            gf2_uint8(
                self.mc.Gp
            )
        )

        return (
            gf2_matmul(
                product,
                G_pub,
            )
        )

    def generate_evaluation_key(
        self,
        tau_F=None,
        batch_size=128,
        max_packed_bytes=512 * 1024 * 1024,
        max_pairs=200000,
        allow_large=False,
        progress=True,
        set_default=True,
    ):
        """
        Generate the compressed public Y_{u,v} table.

        Storage:

            pair_count
                =
            n(n+1)/2

        rows, each containing n packed bits.

        Development guards are enabled by default because evaluation-key
        generation is quadratic in the ciphertext length and the clean
        row construction is computationally expensive.
        """

        self._require_lambda()

        if tau_F is None:
            tau_F = (
                self.tau_F
            )

        tau_F = int(
            tau_F
        )

        batch_size = int(
            batch_size
        )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if not (
            0
            <=
            tau_F
            <=
            self.n
        ):
            raise ValueError(
                "tau_F must satisfy "
                "0 <= tau_F <= n."
            )

        pair_count = (
            fused_pair_count(
                self.n
            )
        )

        row_bytes = (
            (
                self.n
                +
                7
            )
            //
            8
        )

        packed_bytes = (
            pair_count
            *
            row_bytes
        )

        if (
            not allow_large
            and
            pair_count
            >
            int(
                max_pairs
            )
        ):
            raise RuntimeError(
                "Fused evaluation-key pair count "
                f"{pair_count:,} exceeds the current "
                f"development guard {max_pairs:,}. "
                "Use allow_large=True only after profiling."
            )

        if (
            not allow_large
            and
            packed_bytes
            >
            int(
                max_packed_bytes
            )
        ):
            raise RuntimeError(
                "Packed Y table would require "
                f"{packed_bytes / 2**20:.2f} MiB, "
                "exceeding the current development guard."
            )

        Y_packed = np.empty(
            (
                pair_count,
                row_bytes,
            ),
            dtype=np.uint8,
        )

        G_pub = (
            gf2_uint8(
                self.mc.Gp
            )
        )

        write_index = 0

        product_batch = []

        start = (
            time.perf_counter()
        )

        last_percent = -1

        def flush_batch():
            nonlocal write_index,product_batch, last_percent
            if not product_batch:
                return

            products = np.asarray(
                product_batch,
                dtype=np.uint8,
            )

            clean_rows = (
                gf2_matmul(
                    products,
                    G_pub,
                )
            )

            if tau_F > 0:
                for row_index in range(
                    clean_rows.shape[0]
                ):
                    positions = (
                        self.rng.choice(
                            self.n,
                            size=tau_F,
                            replace=False,
                        )
                    )

                    clean_rows[
                        row_index,
                        positions,
                    ] ^= 1

            packed = (
                np.packbits(
                    clean_rows,
                    axis=1,
                    bitorder="little",
                )
            )

            batch_length = int(
                packed.shape[0]
            )

            Y_packed[
                write_index:
                write_index
                +
                batch_length
            ] = packed

            write_index += (
                batch_length
            )

            product_batch = []

            if progress:
                percent = int(
                    100
                    *
                    write_index
                    /
                    pair_count
                )

                if (
                    percent // 10
                    >
                    last_percent // 10
                ):
                    print(
                        "  fused EVK: "
                        f"{write_index:,}/"
                        f"{pair_count:,} rows "
                        f"({percent}%)"
                    )

                    last_percent = (
                        percent
                    )

        for u in range(
            self.n
        ):
            lambda_u = (
                self.Lambda[
                    u
                ]
            )

            for v in range(
                u,
                self.n,
            ):
                product = (
                    truncated_product(
                        lambda_u,
                        self.Lambda[
                            v
                        ],
                        output_length=self.k,
                    )
                )

                product_batch.append(
                    product
                )

                if (
                    len(
                        product_batch
                    )
                    >=
                    batch_size
                ):
                    flush_batch()

        flush_batch()

        generation_seconds = (
            time.perf_counter()
            -
            start
        )

        if (
            write_index
            !=
            pair_count
        ):
            raise RuntimeError(
                "Evaluation-key generation produced "
                f"{write_index} rows; "
                f"expected {pair_count}."
            )

        evk = (
            FusedEvaluationKey(
                n=self.n,
                k=self.k,
                ell=self.ell,
                tau_F=tau_F,
                Y_packed=Y_packed,
                generation_seconds=(
                    generation_seconds
                ),
            )
        )

        if set_default:
            self.evk = (
                evk
            )

        return evk

    # ==================================================================
    # Evaluation-key diagnostics
    # ==================================================================

    def verify_evaluation_key(
        self,
        evk=None,
        samples=32,
    ):
        """
        Secret diagnostic.

        For sampled pairs verify

            wt(
                Y_{u,v} + K_{u,v}
            )
                =
            tau_F.
        """

        self._require_lambda()

        if evk is None:
            evk = (
                self.evk
            )

        if evk is None:
            raise RuntimeError(
                "No evaluation key is available."
            )

        if (
            evk.n != self.n
            or
            evk.k != self.k
        ):
            raise ValueError(
                "Evaluation key does not match "
                "this HE context."
            )

        samples = int(
            samples
        )

        if samples <= 0:
            raise ValueError(
                "samples must be positive."
            )

        observed_weights = []

        for _ in range(
            samples
        ):
            u = int(
                self.rng.integers(
                    0,
                    self.n,
                )
            )

            v = int(
                self.rng.integers(
                    u,
                    self.n,
                )
            )

            K = (
                self.clean_fused_row(
                    u,
                    v,
                )
            )

            Y = (
                evk.get_row(
                    u,
                    v,
                )
            )

            noise = (
                K
                ^
                Y
            )

            observed_weights.append(
                hamming_weight(
                    noise
                )
            )

        all_correct = all(
            weight == evk.tau_F
            for weight
            in observed_weights
        )

        return {
            "samples": (
                samples
            ),
            "expected_tau_F": (
                evk.tau_F
            ),
            "observed_weights": (
                observed_weights
            ),
            "all_rows_have_expected_noise_weight": (
                all_correct
            ),
        }

    # ==================================================================
    # Phase 3D: ciphertext multiplication
    # ==================================================================

    def eval_mult(
        self,
        ciphertext_left,
        ciphertext_right,
        evk=None,
        return_selector_count=False,
    ):
        if evk is None:
            evk = (
                self.evk
            )

        if evk is None:
            raise RuntimeError(
                "No fused evaluation key is available."
            )

        return (
            evk.evaluate_multiply(
                ciphertext_left,
                ciphertext_right,
                return_selector_count=(
                    return_selector_count
                ),
            )
        )

    # ==================================================================
    # Secret multiplication diagnostics
    # ==================================================================

    def multiplication_diagnostics(
        self,
        ciphertext_left,
        ciphertext_right,
        evk=None,
    ):
        """
        Secret diagnostic decomposition.

        Let

            x1 = c1 Lambda,
            x2 = c2 Lambda.

        The clean fused output is

            (x1 odot x2) G_pub.

        Compare the actual Y-based output against this codeword.
        """

        self._require_lambda()

        if evk is None:
            evk = (
                self.evk
            )

        if evk is None:
            raise RuntimeError(
                "No fused evaluation key is available."
            )

        output, selected_count = (
            evk.evaluate_multiply(
                ciphertext_left,
                ciphertext_right,
                return_selector_count=True,
            )
        )

        c1 = gf2_vector(
            ciphertext_left,
            expected_length=self.n,
            name="ciphertext_left",
        )

        c2 = gf2_vector(
            ciphertext_right,
            expected_length=self.n,
            name="ciphertext_right",
        )

        x1 = (
            gf2_matmul(
                c1,
                self.Lambda,
            )
        )

        x2 = (
            gf2_matmul(
                c2,
                self.Lambda,
            )
        )

        product_rep = (
            truncated_product(
                x1,
                x2,
                output_length=self.k,
            )
        )

        G_pub = (
            gf2_uint8(
                self.mc.Gp
            )
        )

        expected_clean = (
            gf2_matmul(
                product_rep,
                G_pub,
            )
        )

        actual = (
            gf2_vector(
                output,
                expected_length=self.n,
                name="evaluated ciphertext",
            )
        )

        fused_noise = (
            actual
            ^
            expected_clean
        )

        fused_noise_weight = (
            hamming_weight(
                fused_noise
            )
        )

        expected_message = (
            truncated_product(
                x1[
                    :self.ell
                ],
                x2[
                    :self.ell
                ],
                output_length=self.ell,
            )
        )

        decode_success = False
        decoded_message = None
        decode_error = None

        try:
            decoded_message = (
                self.decrypt(
                    output
                )
            )

            decode_success = (
                np.array_equal(
                    decoded_message,
                    expected_message,
                )
            )

        except Exception as exc:
            decode_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        return {
            "output": (
                output
            ),
            "selected_rows": (
                selected_count
            ),
            "projected_left": (
                x1
            ),
            "projected_right": (
                x2
            ),
            "product_representative": (
                product_rep
            ),
            "expected_clean": (
                expected_clean
            ),
            "fused_noise": (
                fused_noise
            ),
            "fused_noise_weight": (
                fused_noise_weight
            ),
            "decoder_radius": (
                self.decoder_radius
            ),
            "within_decoder_radius": (
                fused_noise_weight
                <=
                self.decoder_radius
            ),
            "expected_message": (
                expected_message
            ),
            "decoded_message": (
                decoded_message
            ),
            "decode_success": (
                decode_success
            ),
            "decode_error": (
                decode_error
            ),
        }

    # ==================================================================
    # Trusted refresh
    # ==================================================================

    def refresh(
        self,
        ciphertext,
    ):
        """
        Trusted/key-holder-assisted refresh:

            decrypt message
                ->
            fresh lift
                ->
            fresh admissible error
                ->
            re-encrypt.
        """

        plaintext = (
            self.decrypt(
                ciphertext
            )
        )

        return (
            self.encrypt(
                plaintext
            )
        )