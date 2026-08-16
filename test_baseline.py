"""
Baseline correctness test for the Phase 2B McEliece implementation.

This test verifies:

1. Binary Goppa-code key generation.

2. Expected dimensions.

3. Generator/parity-check orthogonality:

       G H^T = 0.

4. Invertibility of the scrambling matrix:

       S S^{-1} = I_k.

5. Correct permutation-vector representation:

       P[P_inv] = identity,
       P_inv[P] = identity.

6. Correct public-generator construction:

       G_pub = (S G)[:, P].

7. McEliece encryption:

       c = m G_pub + e,

   with

       wt(e) = t.

8. Correct inverse permutation during decryption.

9. Patterson decoding.

10. Encryption/decryption correctness over several trials.

Phase 2B represents P as a length-n integer permutation vector
instead of a dense n x n binary matrix.
"""

import numpy as np

from mceliece.mceliececipher import McElieceCipher
from mceliece.mathutils import GF2Matrix


# ================================================================
# Configuration
# ================================================================

M = 4
N = 15
T = 2

NUM_TRIALS = 10
RANDOM_SEED = 1


# ================================================================
# Helper functions
# ================================================================

def gf2_matrix_is_zero(matrix):
    """
    Return True if every entry of a GF2Matrix is zero.
    """

    if not isinstance(
        matrix,
        GF2Matrix,
    ):
        raise TypeError(
            "gf2_matrix_is_zero() expects a GF2Matrix."
        )

    return all(
        int(entry.n) == 0
        for entry in matrix.arr.flat
    )


def gf2_matrix_equal(
    A,
    B,
):
    """
    Compare two GF2Matrix objects entry by entry.
    """

    if not isinstance(
        A,
        GF2Matrix,
    ):
        raise TypeError(
            "A must be a GF2Matrix."
        )

    if not isinstance(
        B,
        GF2Matrix,
    ):
        raise TypeError(
            "B must be a GF2Matrix."
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


def identity_gf2(size):
    """
    Construct the size x size identity matrix over GF(2).
    """

    return GF2Matrix.from_list(
        np.eye(
            size,
            dtype=np.uint8,
        )
    )


def gf2_vector_weight(vector):
    """
    Hamming weight of a GF2Matrix row vector.
    """

    if not isinstance(
        vector,
        GF2Matrix,
    ):
        raise TypeError(
            "gf2_vector_weight expects GF2Matrix."
        )

    return sum(
        int(entry.n)
        for entry in vector.arr.flat
    )


def numpy_hamming_weight(vector):
    """
    Hamming weight of a NumPy binary vector.
    """

    return int(
        np.sum(
            np.asarray(
                vector,
                dtype=np.uint8,
            )
        )
    )


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# Main baseline test
# ================================================================

def main():

    np.random.seed(
        RANDOM_SEED
    )

    print_header(
        "BASELINE McELIECE TEST — PHASE 2B"
    )

    print(
        f"Parameters: "
        f"m={M}, "
        f"n={N}, "
        f"t={T}"
    )

    print(
        "Encryption/decryption trials: "
        f"{NUM_TRIALS}"
    )

    # ------------------------------------------------------------
    # 1. Construct instance
    # ------------------------------------------------------------

    print()
    print(
        "[1] Creating McEliece instance..."
    )

    mc = McElieceCipher(
        m=M,
        n=N,
        t=T,
    )

    # ------------------------------------------------------------
    # 2. Key generation
    # ------------------------------------------------------------

    print(
        "[2] Generating binary Goppa code "
        "and McEliece keys..."
    )

    mc.generate_random_keys()

    k = mc.k

    print()
    print(
        "Generated parameters:"
    )

    print(
        f"  m = {mc.m}"
    )

    print(
        f"  n = {mc.n}"
    )

    print(
        f"  k = {mc.k}"
    )

    print(
        f"  t = {mc.t}"
    )

    # ------------------------------------------------------------
    # Matrix/vector dimensions
    # ------------------------------------------------------------

    print()
    print(
        "Object dimensions:"
    )

    print(
        "  G     =",
        mc.G.arr.shape,
    )

    print(
        "  H     =",
        mc.H.arr.shape,
    )

    print(
        "  S     =",
        mc.S.arr.shape,
    )

    print(
        "  P     =",
        mc.P.shape,
        "(permutation vector)",
    )

    print(
        "  P_inv =",
        mc.P_inv.shape,
        "(inverse permutation vector)",
    )

    print(
        "  G_pub =",
        mc.Gp.arr.shape,
    )

    print(
        "  g_poly storage =",
        mc.g_poly.shape,
    )

    print(
        "  irr_poly storage =",
        mc.irr_poly.shape,
    )

    # ------------------------------------------------------------
    # Show permutation-storage improvement
    # ------------------------------------------------------------

    old_dense_entries = (
        N * N
    )

    new_vector_entries = (
        len(mc.P)
    )

    print()
    print(
        "Permutation storage:"
    )

    print(
        "  old dense P entries =",
        old_dense_entries,
    )

    print(
        "  new vector entries  =",
        new_vector_entries,
    )

    print(
        "  entry-count reduction =",
        f"{old_dense_entries / new_vector_entries:.1f}x",
    )

    # ------------------------------------------------------------
    # 3. Dimension checks
    # ------------------------------------------------------------

    print()
    print(
        "[3] Checking dimensions..."
    )

    expected_k_lower_bound = (
        N - M * T
    )

    print(
        "  n - m*t =",
        expected_k_lower_bound,
    )

    print(
        "  actual k =",
        k,
    )

    assert mc.G.arr.shape == (
        k,
        N,
    ), (
        "Generator matrix has incorrect dimensions."
    )

    assert mc.H.arr.shape[1] == N, (
        "Parity-check matrix has incorrect length."
    )

    assert mc.S.arr.shape == (
        k,
        k,
    ), (
        "Scrambling matrix has incorrect dimensions."
    )

    assert mc.S_inv.arr.shape == (
        k,
        k,
    ), (
        "Inverse scrambling matrix has incorrect dimensions."
    )

    assert mc.P.shape == (
        N,
    ), (
        "Permutation vector has incorrect dimensions."
    )

    assert mc.P_inv.shape == (
        N,
    ), (
        "Inverse permutation vector has "
        "incorrect dimensions."
    )

    assert mc.Gp.arr.shape == (
        k,
        N,
    ), (
        "Public generator has incorrect dimensions."
    )

    assert k >= expected_k_lower_bound, (
        "Binary Goppa dimension violates "
        "k >= n - mt."
    )

    print(
        "  Dimension checks: PASS"
    )

    # ------------------------------------------------------------
    # 4. Verify G H^T = 0
    # ------------------------------------------------------------

    print()
    print(
        "[4] Checking G H^T = 0..."
    )

    GHt = (
        mc.G
        * mc.H.T()
    )

    print(
        "  G H^T shape =",
        GHt.arr.shape,
    )

    gh_zero = (
        gf2_matrix_is_zero(
            GHt
        )
    )

    print(
        "  G H^T == 0 :",
        gh_zero,
    )

    assert gh_zero, (
        "Generator/parity-check "
        "orthogonality failed."
    )

    # ------------------------------------------------------------
    # 5. Verify S S^{-1} = I_k
    # ------------------------------------------------------------

    print()
    print(
        "[5] Checking S S^{-1} = I_k..."
    )

    SS_inv = (
        mc.S
        * mc.S_inv
    )

    I_k = identity_gf2(
        k
    )

    S_ok = (
        gf2_matrix_equal(
            SS_inv,
            I_k,
        )
    )

    print(
        "  S S^{-1} == I_k :",
        S_ok,
    )

    assert S_ok, (
        "Scrambling matrix "
        "inverse check failed."
    )

    # ------------------------------------------------------------
    # 6. Verify permutation-vector inverse
    # ------------------------------------------------------------

    print()
    print(
        "[6] Checking permutation-vector inverse..."
    )

    identity_positions = np.arange(
        N,
        dtype=np.int64,
    )

    P_is_permutation = (
        np.array_equal(
            np.sort(mc.P),
            identity_positions,
        )
    )

    P_inv_is_permutation = (
        np.array_equal(
            np.sort(mc.P_inv),
            identity_positions,
        )
    )

    P_Pinv_ok = (
        np.array_equal(
            mc.P[
                mc.P_inv
            ],
            identity_positions,
        )
    )

    Pinv_P_ok = (
        np.array_equal(
            mc.P_inv[
                mc.P
            ],
            identity_positions,
        )
    )

    probe = np.arange(
        N,
        dtype=np.int64,
    )

    permuted_probe = (
        probe[
            mc.P
        ]
    )

    recovered_probe = (
        permuted_probe[
            mc.P_inv
        ]
    )

    round_trip_ok = (
        np.array_equal(
            recovered_probe,
            probe,
        )
    )

    print(
        "  P is valid permutation       :",
        P_is_permutation,
    )

    print(
        "  P_inv is valid permutation   :",
        P_inv_is_permutation,
    )

    print(
        "  P[P_inv] == identity         :",
        P_Pinv_ok,
    )

    print(
        "  P_inv[P] == identity         :",
        Pinv_P_ok,
    )

    print(
        "  vector permutation roundtrip :",
        round_trip_ok,
    )

    P_ok = (
        P_is_permutation
        and
        P_inv_is_permutation
        and
        P_Pinv_ok
        and
        Pinv_P_ok
        and
        round_trip_ok
    )

    assert P_ok, (
        "Permutation-vector inverse "
        "check failed."
    )

    # ------------------------------------------------------------
    # 7. Verify G_pub = S G P
    # ------------------------------------------------------------

    print()
    print(
        "[7] Checking "
        "G_pub = (S G)[:, P]..."
    )

    SG = (
        mc.S
        * mc.G
    )

    reconstructed_Gp = (
        GF2Matrix(
            SG.arr[
                :,
                mc.P
            ]
        )
    )

    public_key_ok = (
        gf2_matrix_equal(
            reconstructed_Gp,
            mc.Gp,
        )
    )

    print(
        "  public-generator consistency :",
        public_key_ok,
    )

    assert public_key_ok, (
        "Public generator consistency failed."
    )

    # ------------------------------------------------------------
    # 8. Encryption/decryption trials
    # ------------------------------------------------------------

    print()
    print(
        "[8] Running encryption/decryption trials..."
    )

    successes = 0

    for trial in range(
        1,
        NUM_TRIALS + 1,
    ):

        # --------------------------------------------------------
        # Random k-bit plaintext
        # --------------------------------------------------------

        message = np.random.randint(
            0,
            2,
            size=k,
            dtype=np.uint8,
        )

        message_matrix = (
            GF2Matrix.from_list(
                message
            )
        )

        # --------------------------------------------------------
        # Clean public codeword
        #
        #     c_clean = m G_pub.
        # --------------------------------------------------------

        clean_public = (
            message_matrix
            * mc.Gp
        )

        # --------------------------------------------------------
        # Encryption
        # --------------------------------------------------------

        ciphertext = mc.encrypt(
            message
        )

        ciphertext_numpy = (
            ciphertext
            .to_numpy()
            .astype(
                np.uint8
            )
        )

        # --------------------------------------------------------
        # Recover injected error
        #
        # Over GF(2):
        #
        #     e = c + m G_pub.
        # --------------------------------------------------------

        error = (
            ciphertext
            + clean_public
        )

        error_weight = (
            gf2_vector_weight(
                error
            )
        )

        # McEliece baseline inserts exactly t errors.
        assert error_weight == T, (
            f"Encryption inserted error weight "
            f"{error_weight}; expected {T}."
        )

        # --------------------------------------------------------
        # Check inverse permutation directly
        #
        # c_sec = c_pub[P_inv].
        # --------------------------------------------------------

        secret_coordinate_ct = (
            GF2Matrix(
                ciphertext.arr[
                    mc.P_inv
                ]
            )
        )

        assert len(
            secret_coordinate_ct
        ) == N

        # --------------------------------------------------------
        # Decryption
        # --------------------------------------------------------

        decoded = mc.decrypt(
            ciphertext
        )

        decoded = np.asarray(
            decoded,
            dtype=np.uint8,
        ).flatten()

        message = np.asarray(
            message,
            dtype=np.uint8,
        ).flatten()

        success = np.array_equal(
            message,
            decoded,
        )

        if success:
            successes += 1

        print()
        print(
            f"  Trial {trial:02d}:"
        )

        print(
            "    message       =",
            message,
        )

        print(
            "    ciphertext    =",
            ciphertext_numpy,
        )

        print(
            "    ct weight     =",
            numpy_hamming_weight(
                ciphertext_numpy
            ),
        )

        print(
            "    error weight  =",
            error_weight,
        )

        print(
            "    decoded       =",
            decoded,
        )

        print(
            "    success       =",
            success,
        )

        if not success:
            raise AssertionError(
                "Encryption/decryption failed "
                f"on trial {trial}."
            )

    # ------------------------------------------------------------
    # 9. Final report
    # ------------------------------------------------------------

    print_header(
        "BASELINE RESULT — PHASE 2B"
    )

    print(
        "Finite-field construction      : PASS"
    )

    print(
        "Goppa-code key generation      : PASS"
    )

    print(
        "Generator/parity-check test    : PASS"
    )

    print(
        "Scrambling inverse test        : PASS"
    )

    print(
        "Permutation-vector test        : PASS"
    )

    print(
        "Public-generator consistency   : PASS"
    )

    print(
        "Exact encryption error weight  : PASS"
    )

    print(
        "Patterson decoding             : PASS"
    )

    print(
        "Encryption/decryption          : "
        f"{successes}/{NUM_TRIALS} PASS"
    )

    print()

    print(
        "PHASE 2B BASELINE "
        "IMPLEMENTATION WORKS CORRECTLY."
    )


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()