"""
Scaling regression for the Phase 2B binary-Goppa McEliece implementation.

This test verifies, for several parameter sets:

1. Binary Goppa-code construction.
2. Dimension and rank conditions.
3. G H^T = 0.
4. S S^{-1} = I_k.
5. Vector representation of the secret permutation.
6. Correct inverse permutation.
7. G_pub = (S G)[:, P].
8. Exact McEliece encryption error weight.
9. Patterson decoding.
10. Encryption/decryption correctness.
11. Basic timing measurements.

Phase 2B stores the secret permutation as

    P     : ndarray of shape (n,)
    P_inv : ndarray of shape (n,)

instead of dense n x n matrices.
"""

import time

import numpy as np

from mceliece.mceliececipher import McElieceCipher
from mceliece.mathutils import GF2Matrix


# ================================================================
# Configuration
# ================================================================

PARAMETER_SETS = [
    # m, n, t, trials
    (4, 15, 2, 5),
    (5, 31, 2, 5),
    (6, 63, 3, 5),
]

BASE_SEED = 20260816


# ================================================================
# Helper functions
# ================================================================

def gf2_numpy(M):
    """
    Convert GF2Matrix to ordinary uint8 NumPy array
    while preserving its shape.
    """

    if not isinstance(
        M,
        GF2Matrix,
    ):
        raise TypeError(
            "gf2_numpy expects GF2Matrix."
        )

    return np.array(
        [
            int(x.n)
            for x in M.arr.flat
        ],
        dtype=np.uint8,
    ).reshape(
        M.arr.shape
    )


def matrix_is_zero(M):
    """
    Test whether a GF2Matrix is zero.
    """

    return not np.any(
        gf2_numpy(M)
    )


def matrix_equal(
    A,
    B,
):
    """
    Compare two GF2Matrix objects.
    """

    return np.array_equal(
        gf2_numpy(A),
        gf2_numpy(B),
    )


def gf2_identity(size):
    """
    Identity matrix over GF(2).
    """

    return GF2Matrix.from_list(
        np.eye(
            size,
            dtype=np.uint8,
        )
    )


def vector_weight(M):
    """
    Hamming weight of a GF2Matrix row vector.
    """

    if not isinstance(
        M,
        GF2Matrix,
    ):
        raise TypeError(
            "vector_weight expects GF2Matrix."
        )

    return sum(
        int(x.n)
        for x in M.arr.flat
    )


def validate_permutation(
    P,
    P_inv,
    n,
):
    """
    Validate a permutation vector and its inverse.

    Required identities:

        P[P_inv] = id,

        P_inv[P] = id.

    We also check the induced action on a probe vector.
    """

    P = np.asarray(
        P,
        dtype=np.int64,
    ).reshape(-1)

    P_inv = np.asarray(
        P_inv,
        dtype=np.int64,
    ).reshape(-1)

    identity = np.arange(
        n,
        dtype=np.int64,
    )

    shape_ok = (
        P.shape == (n,)
        and
        P_inv.shape == (n,)
    )

    if not shape_ok:
        return False

    P_valid = np.array_equal(
        np.sort(P),
        identity,
    )

    P_inv_valid = np.array_equal(
        np.sort(P_inv),
        identity,
    )

    left_inverse = np.array_equal(
        P[P_inv],
        identity,
    )

    right_inverse = np.array_equal(
        P_inv[P],
        identity,
    )

    probe = np.arange(
        n,
        dtype=np.int64,
    )

    round_trip = np.array_equal(
        probe[P][P_inv],
        probe,
    )

    return (
        P_valid
        and
        P_inv_valid
        and
        left_inverse
        and
        right_inverse
        and
        round_trip
    )


# ================================================================
# Single parameter-set test
# ================================================================

def test_parameter_set(
    m,
    n,
    t,
    trials,
    seed,
):
    print()
    print("=" * 78)

    print(
        f"SCALING TEST: "
        f"m={m}, n={n}, t={t}"
    )

    print("=" * 78)

    np.random.seed(
        seed
    )

    # ------------------------------------------------------------
    # 1. Parameter checks
    # ------------------------------------------------------------

    dimension_lower_bound = (
        n - m * t
    )

    print()
    print(
        "[1] Parameter checks"
    )

    print(
        f"  field size             = "
        f"{2 ** m}"
    )

    print(
        f"  nonzero field elements = "
        f"{2 ** m - 1}"
    )

    print(
        f"  code length n          = "
        f"{n}"
    )

    print(
        f"  Goppa degree t         = "
        f"{t}"
    )

    print(
        "  dimension lower bound  = "
        f"n - m*t = "
        f"{dimension_lower_bound}"
    )

    if n > 2 ** m - 1:
        raise AssertionError(
            "Current Goppa backend uses the "
            "nonzero support "
            "{1, alpha, ..., alpha^(n-1)}, "
            "so n must satisfy "
            "n <= 2^m - 1."
        )

    # ------------------------------------------------------------
    # 2. Key generation
    # ------------------------------------------------------------

    print()
    print(
        "[2] Generating Goppa code "
        "and McEliece keys..."
    )

    mc = McElieceCipher(
        m=m,
        n=n,
        t=t,
    )

    start = time.perf_counter()

    mc.generate_random_keys()

    keygen_time = (
        time.perf_counter()
        - start
    )

    k = mc.k

    print()
    print(
        "Generated parameters:"
    )

    print(
        f"  m = {m}"
    )

    print(
        f"  n = {n}"
    )

    print(
        f"  k = {k}"
    )

    print(
        f"  t = {t}"
    )

    print()
    print(
        "Object dimensions:"
    )

    print(
        f"  G     = "
        f"{mc.G.arr.shape}"
    )

    print(
        f"  H     = "
        f"{mc.H.arr.shape}"
    )

    print(
        f"  S     = "
        f"{mc.S.arr.shape}"
    )

    print(
        f"  P     = "
        f"{mc.P.shape} "
        f"(permutation vector)"
    )

    print(
        f"  P_inv = "
        f"{mc.P_inv.shape} "
        f"(inverse permutation)"
    )

    print(
        f"  G_pub = "
        f"{mc.Gp.arr.shape}"
    )

    # ------------------------------------------------------------
    # Permutation storage comparison
    # ------------------------------------------------------------

    dense_P_entries = (
        n * n
    )

    vector_P_entries = (
        n
    )

    print()
    print(
        "Permutation storage:"
    )

    print(
        f"  dense matrix entries   = "
        f"{dense_P_entries}"
    )

    print(
        f"  vector entries         = "
        f"{vector_P_entries}"
    )

    print(
        f"  entry-count reduction  = "
        f"{dense_P_entries / vector_P_entries:.1f}x"
    )

    print()
    print(
        "Key-generation time = "
        f"{keygen_time:.6f} seconds"
    )

    # ------------------------------------------------------------
    # 3. Dimension/rank checks
    # ------------------------------------------------------------

    print()
    print(
        "[3] Checking dimensions and rank..."
    )

    expected_H_shape = (
        m * t,
        n,
    )

    if (
        mc.H.arr.shape
        != expected_H_shape
    ):
        raise AssertionError(
            f"H has shape "
            f"{mc.H.arr.shape}, "
            f"expected "
            f"{expected_H_shape}."
        )

    if (
        mc.G.arr.shape
        != (k, n)
    ):
        raise AssertionError(
            "Unexpected generator "
            "matrix dimensions."
        )

    if (
        mc.Gp.arr.shape
        != (k, n)
    ):
        raise AssertionError(
            "Unexpected public-generator "
            "dimensions."
        )

    if (
        mc.S.arr.shape
        != (k, k)
    ):
        raise AssertionError(
            "Unexpected scrambling-matrix "
            "dimensions."
        )

    if (
        mc.P.shape
        != (n,)
    ):
        raise AssertionError(
            "Unexpected permutation-vector "
            "dimensions."
        )

    if (
        mc.P_inv.shape
        != (n,)
    ):
        raise AssertionError(
            "Unexpected inverse-permutation "
            "dimensions."
        )

    _, rank_H = (
        mc.H.rref()
    )

    print(
        f"  rank(H)                = "
        f"{rank_H}"
    )

    print(
        f"  n - rank(H)            = "
        f"{n - rank_H}"
    )

    print(
        f"  actual k               = "
        f"{k}"
    )

    print(
        f"  Goppa lower bound      = "
        f"{dimension_lower_bound}"
    )

    if k != (
        n - rank_H
    ):
        raise AssertionError(
            "Generator dimension does not "
            "equal nullity of H."
        )

    # Binary Goppa dimension bound:
    #
    #     k >= n - m t.
    #
    # Equality is common for our current tests,
    # but the regression only requires the bound.

    if k < dimension_lower_bound:
        raise AssertionError(
            "Code dimension violates "
            "k >= n - mt."
        )

    print(
        "  Dimension/rank checks: PASS"
    )

    # ------------------------------------------------------------
    # 4. G H^T = 0
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
        f"  G H^T shape = "
        f"{GHt.arr.shape}"
    )

    orthogonal = (
        matrix_is_zero(
            GHt
        )
    )

    print(
        f"  G H^T == 0 : "
        f"{orthogonal}"
    )

    if not orthogonal:
        raise AssertionError(
            "G H^T != 0."
        )

    # ------------------------------------------------------------
    # 5. Scrambling inverse
    # ------------------------------------------------------------

    print()
    print(
        "[5] Checking S S^{-1} = I_k..."
    )

    SSinv = (
        mc.S
        * mc.S_inv
    )

    scrambling_ok = (
        matrix_equal(
            SSinv,
            gf2_identity(k),
        )
    )

    print(
        "  scrambling inverse : "
        f"{scrambling_ok}"
    )

    if not scrambling_ok:
        raise AssertionError(
            "S S^{-1} != I_k."
        )

    # ------------------------------------------------------------
    # 6. Permutation vectors
    # ------------------------------------------------------------

    print()
    print(
        "[6] Checking permutation-vector inverse..."
    )

    identity_positions = np.arange(
        n,
        dtype=np.int64,
    )

    P_valid = np.array_equal(
        np.sort(mc.P),
        identity_positions,
    )

    P_inv_valid = np.array_equal(
        np.sort(mc.P_inv),
        identity_positions,
    )

    P_Pinv_ok = np.array_equal(
        mc.P[
            mc.P_inv
        ],
        identity_positions,
    )

    Pinv_P_ok = np.array_equal(
        mc.P_inv[
            mc.P
        ],
        identity_positions,
    )

    probe = np.arange(
        n,
        dtype=np.int64,
    )

    round_trip_ok = np.array_equal(
        probe[
            mc.P
        ][
            mc.P_inv
        ],
        probe,
    )

    permutation_ok = (
        validate_permutation(
            mc.P,
            mc.P_inv,
            n,
        )
    )

    print(
        "  P is valid permutation       : "
        f"{P_valid}"
    )

    print(
        "  P_inv is valid permutation   : "
        f"{P_inv_valid}"
    )

    print(
        "  P[P_inv] == identity         : "
        f"{P_Pinv_ok}"
    )

    print(
        "  P_inv[P] == identity         : "
        f"{Pinv_P_ok}"
    )

    print(
        "  vector round trip            : "
        f"{round_trip_ok}"
    )

    print(
        "  permutation-vector test      : "
        f"{permutation_ok}"
    )

    if not permutation_ok:
        raise AssertionError(
            "Permutation-vector "
            "inverse failed."
        )

    # ------------------------------------------------------------
    # 7. Public generator
    #
    #       G_pub = S G P
    #
    # With vector P:
    #
    #       G_pub = (S G)[:, P].
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

    expected_Gp = (
        GF2Matrix(
            SG.arr[
                :,
                mc.P
            ]
        )
    )

    public_generator_ok = (
        matrix_equal(
            mc.Gp,
            expected_Gp,
        )
    )

    print(
        "  public-generator consistency : "
        f"{public_generator_ok}"
    )

    if not public_generator_ok:
        raise AssertionError(
            "G_pub != (S G)[:, P]."
        )

    # ------------------------------------------------------------
    # 8. Encryption/decryption trials
    # ------------------------------------------------------------

    print()
    print(
        f"[8] Running {trials} "
        "encryption/decryption trials..."
    )

    successful_trials = 0

    encryption_times = []
    decryption_times = []

    for trial in range(
        1,
        trials + 1,
    ):
        # --------------------------------------------------------
        # Random plaintext
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

        start = time.perf_counter()

        ciphertext = mc.encrypt(
            message
        )

        encryption_time = (
            time.perf_counter()
            - start
        )

        encryption_times.append(
            encryption_time
        )

        # --------------------------------------------------------
        # Recover injected error
        #
        #     e = c + m G_pub
        #
        # over GF(2).
        # --------------------------------------------------------

        error = (
            ciphertext
            + clean_public
        )

        error_weight = (
            vector_weight(
                error
            )
        )

        if error_weight != t:
            raise AssertionError(
                "Encryption inserted error "
                f"weight {error_weight}; "
                f"expected {t}."
            )

        # --------------------------------------------------------
        # Explicitly check inverse permutation
        # --------------------------------------------------------

        secret_coordinate_ct = (
            GF2Matrix(
                ciphertext.arr[
                    mc.P_inv
                ]
            )
        )

        if len(
            secret_coordinate_ct
        ) != n:
            raise AssertionError(
                "Inverse permutation changed "
                "ciphertext length."
            )

        # --------------------------------------------------------
        # Decrypt
        # --------------------------------------------------------

        start = time.perf_counter()

        decoded = mc.decrypt(
            ciphertext
        )

        decryption_time = (
            time.perf_counter()
            - start
        )

        decryption_times.append(
            decryption_time
        )

        message_int = np.asarray(
            message,
            dtype=np.uint8,
        ).reshape(-1)

        decoded_int = np.asarray(
            decoded,
            dtype=np.uint8,
        ).reshape(-1)

        success = np.array_equal(
            message_int,
            decoded_int,
        )

        if success:
            successful_trials += 1

        print()
        print(
            f"  Trial {trial:02d}:"
        )

        print(
            "    injected error weight = "
            f"{error_weight}"
        )

        print(
            "    encryption time       = "
            f"{encryption_time:.6f} s"
        )

        print(
            "    decryption time       = "
            f"{decryption_time:.6f} s"
        )

        print(
            "    success               = "
            f"{success}"
        )

        if not success:
            print(
                "    message = "
                f"{message_int}"
            )

            print(
                "    decoded = "
                f"{decoded_int}"
            )

            raise AssertionError(
                "Decryption failed for "
                f"(m,n,t)=({m},{n},{t})."
            )

    # ------------------------------------------------------------
    # 9. Timing summary
    # ------------------------------------------------------------

    avg_enc = float(
        np.mean(
            encryption_times
        )
    )

    avg_dec = float(
        np.mean(
            decryption_times
        )
    )

    # ------------------------------------------------------------
    # Parameter-set result
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "PARAMETER-SET RESULT"
    )
    print("-" * 78)

    print(
        "Parameters               : "
        f"({m}, {n}, {t})"
    )

    print(
        "Dimension k              : "
        f"{k}"
    )

    print(
        "rank(H)                  : "
        f"{rank_H}"
    )

    print(
        "Permutation representation: "
        f"vector length {n}"
    )

    print(
        "Permutation reduction    : "
        f"{n:.1f}x fewer entries"
    )

    print(
        "Key-generation time      : "
        f"{keygen_time:.6f} s"
    )

    print(
        "Average encryption time  : "
        f"{avg_enc:.6f} s"
    )

    print(
        "Average decryption time  : "
        f"{avg_dec:.6f} s"
    )

    print(
        "Encryption/decryption    : "
        f"{successful_trials}/"
        f"{trials} PASS"
    )

    return {
        "m": m,
        "n": n,
        "t": t,
        "k": k,
        "rank_H": rank_H,
        "keygen_time": keygen_time,
        "avg_enc": avg_enc,
        "avg_dec": avg_dec,
        "successes": successful_trials,
        "trials": trials,
        "permutation_entries": n,
        "dense_permutation_entries": n * n,
    }


# ================================================================
# Main
# ================================================================

def main():
    print()

    print("=" * 78)

    print(
        "BINARY GOPPA / McELIECE "
        "SCALING REGRESSION — PHASE 2B"
    )

    print("=" * 78)

    results = []

    for index, parameters in enumerate(
        PARAMETER_SETS
    ):
        (
            m,
            n,
            t,
            trials,
        ) = parameters

        result = (
            test_parameter_set(
                m=m,
                n=n,
                t=t,
                trials=trials,
                seed=BASE_SEED + index,
            )
        )

        results.append(
            result
        )

    # ------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "FINAL SCALING SUMMARY — PHASE 2B"
    )

    print("=" * 78)

    for result in results:
        print(
            f"(m,n,t)="
            f"({result['m']},"
            f"{result['n']},"
            f"{result['t']})"
            f"  "
            f"k={result['k']}"
            f"  "
            f"rank(H)={result['rank_H']}"
            f"  "
            f"P={result['permutation_entries']}"
            f" entries"
            f"  "
            f"keygen={result['keygen_time']:.4f}s"
            f"  "
            f"dec={result['avg_dec']:.4f}s"
            f"  "
            f"{result['successes']}/"
            f"{result['trials']} PASS"
        )

    print()
    print(
        "Goppa construction           : PASS"
    )

    print(
        "Permutation-vector migration : PASS"
    )

    print(
        "Public-generator consistency : PASS"
    )

    print(
        "Exact encryption error weight: PASS"
    )

    print(
        "Patterson decoding           : PASS"
    )

    print()

    print(
        "ALL PHASE 2B SCALING "
        "REGRESSION TESTS PASSED."
    )


if __name__ == "__main__":
    main()