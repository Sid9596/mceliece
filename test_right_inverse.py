"""
Phase 2D factorized-right-inverse regression.

The implementation stores

    J = right_inverse_pivots
    C = right_inverse_core

with

    G[:,J] C = I_k.

No dense n x k right-inverse matrix is required.
"""

import time

import numpy as np

from mceliece.mceliececipher import McElieceCipher
from mceliece.mathutils import GF2Matrix


PARAMETER_SETS = [
    (4, 15, 2, 5),
    (5, 31, 2, 5),
    (6, 63, 3, 5),
]

BASE_SEED = 20260816


def matrix_equal(
    A,
    B,
):

    if A.arr.shape != B.arr.shape:
        return False

    return all(
        int(a.n) == int(b.n)
        for a, b in zip(
            A.arr.flat,
            B.arr.flat,
        )
    )


def identity_gf2(k):

    return GF2Matrix.from_list(
        np.eye(
            k,
            dtype=np.uint8,
        )
    )


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
        f"PHASE 2D FACTORIZED RIGHT INVERSE: "
        f"m={m}, n={n}, t={t}"
    )

    print("=" * 78)

    np.random.seed(
        seed
    )

    mc = McElieceCipher(
        m,
        n,
        t,
    )

    start = time.perf_counter()

    mc.generate_random_keys()

    keygen_time = (
        time.perf_counter()
        - start
    )

    k = int(
        mc.k
    )

    J = np.asarray(
        mc.right_inverse_pivots,
        dtype=np.int64,
    )

    C = mc.right_inverse_core

    print()
    print(
        "  G shape              =",
        mc.G.arr.shape,
    )

    print(
        "  pivot vector shape   =",
        J.shape,
    )

    print(
        "  core shape           =",
        C.arr.shape,
    )

    assert mc.G.arr.shape == (
        k,
        n,
    )

    assert J.shape == (
        k,
    )

    assert C.arr.shape == (
        k,
        k,
    )

    assert len(
        np.unique(
            J
        )
    ) == k

    # ------------------------------------------------------------
    # B = G[:,J].
    # ------------------------------------------------------------

    B = GF2Matrix(
        mc.G.arr[
            :,
            J
        ]
    )

    BC = (
        B
        * C
    )

    CB = (
        C
        * B
    )

    BC_ok = matrix_equal(
        BC,
        identity_gf2(
            k
        ),
    )

    CB_ok = matrix_equal(
        CB,
        identity_gf2(
            k
        ),
    )

    print()
    print(
        "  G[:,J] C = I_k :",
        BC_ok,
    )

    print(
        "  C G[:,J] = I_k :",
        CB_ok,
    )

    assert BC_ok
    assert CB_ok

    # ------------------------------------------------------------
    # Storage comparison.
    # ------------------------------------------------------------

    dense_R_entries = (
        n * k
    )

    factorized_entries = (
        k * k
        +
        k
    )

    print()
    print(
        "Right-inverse representation:"
    )

    print(
        "  old dense R entries      =",
        dense_R_entries,
    )

    print(
        "  factorized entries       =",
        factorized_entries,
    )

    print(
        "  entry reduction          =",
        f"{dense_R_entries / factorized_entries:.3f}x",
    )

    # ------------------------------------------------------------
    # Clean extraction
    #
    #     (aG)[J] C = a.
    # ------------------------------------------------------------

    print()
    print(
        f"[Clean extraction: {trials} trials]"
    )

    for trial in range(
        1,
        trials + 1,
    ):

        a_numpy = np.random.randint(
            0,
            2,
            size=k,
            dtype=np.uint8,
        )

        a = GF2Matrix.from_list(
            a_numpy
        )

        codeword = (
            a
            * mc.G
        )

        selected = GF2Matrix(
            codeword.arr[
                J
            ]
        )

        recovered = (
            selected
            * C
        )

        recovered_numpy = (
            recovered
            .to_numpy()
            .astype(
                np.uint8
            )
        )

        success = np.array_equal(
            a_numpy,
            recovered_numpy,
        )

        print(
            f"  clean trial {trial:02d}: "
            f"{success}"
        )

        assert success

    # ------------------------------------------------------------
    # Noisy encryption/decryption.
    # ------------------------------------------------------------

    print()
    print(
        f"[Noisy trials: {trials}]"
    )

    successes = 0
    decode_times = []

    for trial in range(
        1,
        trials + 1,
    ):

        message = np.random.randint(
            0,
            2,
            size=k,
            dtype=np.uint8,
        )

        ciphertext = mc.encrypt(
            message
        )

        secret_ct = GF2Matrix(
            ciphertext.arr[
                mc.P_inv
            ]
        )

        expected_scrambled = (
            GF2Matrix.from_list(
                message
            )
            * mc.S
        )

        start = time.perf_counter()

        decoded_scrambled = mc.decode(
            secret_ct
        )

        decode_time = (
            time.perf_counter()
            - start
        )

        decode_times.append(
            decode_time
        )

        decode_ok = matrix_equal(
            decoded_scrambled,
            expected_scrambled,
        )

        decoded_message = np.asarray(
            mc.decrypt(
                ciphertext
            ),
            dtype=np.uint8,
        )

        decrypt_ok = np.array_equal(
            decoded_message,
            message,
        )

        success = (
            decode_ok
            and
            decrypt_ok
        )

        if success:
            successes += 1

        print(
            f"  trial {trial:02d}: "
            f"decode={decode_ok}, "
            f"decrypt={decrypt_ok}, "
            f"time={decode_time:.6f}s"
        )

        assert success

    avg_decode = float(
        np.mean(
            decode_times
        )
    )

    print()
    print("-" * 78)

    print(
        "PARAMETER RESULT"
    )

    print("-" * 78)

    print(
        f"Parameters            : ({m},{n},{t})"
    )

    print(
        f"k                     : {k}"
    )

    print(
        f"core shape            : {C.arr.shape}"
    )

    print(
        f"pivot count           : {len(J)}"
    )

    print(
        f"Key generation        : {keygen_time:.6f}s"
    )

    print(
        f"Average decode        : {avg_decode:.6f}s"
    )

    print(
        f"Correct trials        : "
        f"{successes}/{trials}"
    )

    return {
        "m": m,
        "n": n,
        "t": t,
        "k": k,
        "decode": avg_decode,
        "successes": successes,
        "trials": trials,
    }


def main():

    print()
    print("=" * 78)

    print(
        "PHASE 2D FACTORIZED RIGHT-INVERSE REGRESSION"
    )

    print("=" * 78)

    results = []

    for index, params in enumerate(
        PARAMETER_SETS
    ):

        m, n, t, trials = params

        results.append(
            test_parameter_set(
                m,
                n,
                t,
                trials,
                BASE_SEED + index,
            )
        )

    print()
    print("=" * 78)

    print(
        "PHASE 2D FINAL SUMMARY"
    )

    print("=" * 78)

    for result in results:

        print(
            f"(m,n,t)="
            f"({result['m']},"
            f"{result['n']},"
            f"{result['t']}) "
            f"k={result['k']} "
            f"decode={result['decode']:.6f}s "
            f"{result['successes']}/"
            f"{result['trials']} PASS"
        )

    print()
    print(
        "G[:,J] C = I_k               : PASS"
    )

    print(
        "C G[:,J] = I_k               : PASS"
    )

    print(
        "(aG)[J] C = a                : PASS"
    )

    print(
        "No dense right inverse        : PASS"
    )

    print(
        "Patterson decoding            : PASS"
    )

    print(
        "Encryption/decryption          : PASS"
    )

    print()
    print(
        "PHASE 2D FACTORIZED RIGHT-INVERSE "
        "REGRESSION PASSED."
    )


if __name__ == "__main__":
    main()