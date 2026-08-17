"""
Phase 3 regression for the fused McEliece HE layer.

Tests:

    1. Randomized HE right inverse:
           G R_HE = I_k.

    2. Correct permutation orientation:
           G_pub Lambda = I_k.

    3. Symmetric compressed Y_{u,v} table.

    4. Fused-row equation:
           Y_{u,v}
             =
           (Lambda_u odot Lambda_v) G_pub
           + z_{u,v}.

    5. HE encryption/decryption.

    6. Homomorphic addition.

    7. Homomorphic multiplication with tau_F = 0.

    8. Public evaluation-key serialization.

    9. Diagnostic experiment with tau_F = 1 showing
       actual fused-noise accumulation.
"""

import tempfile
from pathlib import Path

import numpy as np

from mceliece.mceliececipher import (
    McElieceCipher,
)

from mceliece.fused_he import (
    FusedEvaluationKey,
    FusedMcElieceHE,
    fused_pair_count,
    gf2_matmul,
    gf2_uint8,
)


def bits_equal(
    left,
    right,
):
    return np.array_equal(
        np.asarray(
            left,
            dtype=np.uint8,
        ).reshape(-1),
        np.asarray(
            right,
            dtype=np.uint8,
        ).reshape(-1),
    )


def run_parameter_set(
    m,
    n,
    t,
    ell,
    seed,
    multiplication_trials,
):
    print()
    print("=" * 88)

    print(
        "PHASE 3 FUSED-HE TEST"
    )

    print(
        f"(m,n,t,ell) = "
        f"({m},{n},{t},{ell})"
    )

    print("=" * 88)

    # McEliece backend currently uses the global NumPy RNG.
    np.random.seed(
        seed
    )

    mc = McElieceCipher(
        m,
        n,
        t,
    )

    mc.generate_random_keys()

    k = int(
        mc.k
    )

    print()
    print(
        f"Derived code dimension k = {k}"
    )

    # ==========================================================
    # Secret HE context.
    #
    # Main exact multiplication tests use tau_enc = 0 first.
    # Admissible nonzero errors are tested separately.
    # ==========================================================

    he = FusedMcElieceHE(
        mc,
        ell=ell,
        tau_enc=0,
        tau_F=0,
        seed=seed + 1000,
    )

    Lambda = he.construct_transport(
        randomization_trials=64,
        store_right_inverse=True,
    )

    diagnostics = (
        he.transport_diagnostics
    )

    # ==========================================================
    # 1. G R_HE = I_k
    # ==========================================================

    G = gf2_uint8(
        mc.G
    )

    R_he = (
        he.he_right_inverse
    )

    identity = np.eye(
        k,
        dtype=np.uint8,
    )

    right_inverse_ok = (
        np.array_equal(
            gf2_matmul(
                G,
                R_he,
            ),
            identity,
        )
    )

    print()
    print(
        "[Phase 3A]"
    )

    print(
        "G R_HE = I_k                    : "
        f"{'PASS' if right_inverse_ok else 'FAIL'}"
    )

    if not right_inverse_ok:
        raise AssertionError(
            "Randomized HE right inverse failed."
        )

    # ==========================================================
    # 2. G_pub Lambda = I_k
    # ==========================================================

    G_pub = gf2_uint8(
        mc.Gp
    )

    transport_ok = (
        np.array_equal(
            gf2_matmul(
                G_pub,
                Lambda,
            ),
            identity,
        )
    )

    print(
        "G_pub Lambda = I_k              : "
        f"{'PASS' if transport_ok else 'FAIL'}"
    )

    if not transport_ok:
        raise AssertionError(
            "Transport identity failed."
        )

    print(
        "Sparse R0 zero rows              : "
        f"{diagnostics['base_zero_rows']}"
    )

    print(
        "Randomized R_HE zero rows        : "
        f"{diagnostics['randomized_zero_rows']}"
    )

    print(
        "Lambda zero rows                 : "
        f"{diagnostics['lambda_zero_rows']}"
    )

    print(
        "Lambda row-weight range          : "
        f"[{diagnostics['lambda_row_weight_min']}, "
        f"{diagnostics['lambda_row_weight_max']}]"
    )

    print(
        "Lambda mean row weight           : "
        f"{diagnostics['lambda_row_weight_mean']:.3f}"
    )

    # ==========================================================
    # 3. Clean fused table tau_F = 0
    # ==========================================================

    print()
    print(
        "[Phase 3B/3C]"
    )

    evk = he.generate_evaluation_key(
        tau_F=0,
        batch_size=64,
        max_pairs=200000,
        progress=False,
    )

    expected_pairs = (
        fused_pair_count(
            n
        )
    )

    pair_count_ok = (
        evk.pair_count
        ==
        expected_pairs
    )

    print(
        "Compressed pair count            : "
        f"{evk.pair_count}"
    )

    print(
        "Expected n(n+1)/2                : "
        f"{expected_pairs}"
    )

    print(
        "Pair-count check                 : "
        f"{'PASS' if pair_count_ok else 'FAIL'}"
    )

    if not pair_count_ok:
        raise AssertionError(
            "Wrong fused pair count."
        )

    print(
        "Packed Y bytes                   : "
        f"{evk.packed_bytes}"
    )

    print(
        "Equivalent uint8 Y bytes         : "
        f"{evk.unpacked_uint8_bytes}"
    )

    print(
        "EVK generation time              : "
        f"{evk.generation_seconds:.6f} s"
    )

    verification = (
        he.verify_evaluation_key(
            evk,
            samples=min(
                32,
                evk.pair_count,
            ),
        )
    )

    fused_rows_ok = (
        verification[
            "all_rows_have_expected_noise_weight"
        ]
    )

    print(
        "Sampled Y = K + z verification   : "
        f"{'PASS' if fused_rows_ok else 'FAIL'}"
    )

    if not fused_rows_ok:
        raise AssertionError(
            "Fused-row construction failed."
        )

    # ==========================================================
    # 4. Public EVK serialization
    # ==========================================================

    print()
    print(
        "[Public fused-key serialization]"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        evk_path = (
            Path(
                temp_dir
            )
            /
            "fused_evk.npz"
        )

        evk.save(
            evk_path
        )

        loaded_evk = (
            FusedEvaluationKey.load(
                evk_path
            )
        )

        serialization_ok = (
            loaded_evk.n
            ==
            evk.n
            and
            loaded_evk.k
            ==
            evk.k
            and
            loaded_evk.ell
            ==
            evk.ell
            and
            loaded_evk.tau_F
            ==
            evk.tau_F
            and
            np.array_equal(
                loaded_evk.Y_packed,
                evk.Y_packed,
            )
        )

    print(
        "Save/load Y_packed               : "
        f"{'PASS' if serialization_ok else 'FAIL'}"
    )

    if not serialization_ok:
        raise AssertionError(
            "Evaluation-key serialization failed."
        )

    he.evk = (
        loaded_evk
    )

    # ==========================================================
    # 5. HE encryption/decryption with clean physical noise
    # ==========================================================

    print()
    print(
        "[HE encryption/decryption]"
    )

    encryption_successes = 0

    for _ in range(
        10
    ):
        mu = he.rng.integers(
            0,
            2,
            size=ell,
            dtype=np.uint8,
        )

        ciphertext = (
            he.encrypt(
                mu
            )
        )

        decoded = (
            he.decrypt(
                ciphertext
            )
        )

        if bits_equal(
            decoded,
            mu,
        ):
            encryption_successes += 1

    print(
        "Clean HE encryption/decryption   : "
        f"{encryption_successes}/10"
    )

    if encryption_successes != 10:
        raise AssertionError(
            "HE encryption/decryption failed."
        )

    # ==========================================================
    # 6. Homomorphic addition
    # ==========================================================

    addition_successes = 0

    for _ in range(
        10
    ):
        mu1 = he.rng.integers(
            0,
            2,
            size=ell,
            dtype=np.uint8,
        )

        mu2 = he.rng.integers(
            0,
            2,
            size=ell,
            dtype=np.uint8,
        )

        c1 = he.encrypt(
            mu1
        )

        c2 = he.encrypt(
            mu2
        )

        c_add = (
            he.eval_add(
                c1,
                c2,
            )
        )

        decoded = (
            he.decrypt(
                c_add
            )
        )

        expected = (
            he.plaintext_add(
                mu1,
                mu2,
            )
        )

        if bits_equal(
            decoded,
            expected,
        ):
            addition_successes += 1

    print(
        "Homomorphic addition             : "
        f"{addition_successes}/10"
    )

    if addition_successes != 10:
        raise AssertionError(
            "Homomorphic addition failed."
        )

    # ==========================================================
    # 7. Exact fused multiplication, tau_F = 0
    # ==========================================================

    print()
    print(
        "[Phase 3D exact multiplication]"
    )

    multiplication_successes = 0

    selector_counts = []

    for _ in range(
        multiplication_trials
    ):
        mu1 = he.rng.integers(
            0,
            2,
            size=ell,
            dtype=np.uint8,
        )

        mu2 = he.rng.integers(
            0,
            2,
            size=ell,
            dtype=np.uint8,
        )

        c1 = he.encrypt(
            mu1
        )

        c2 = he.encrypt(
            mu2
        )

        c_mul, selected = (
            he.eval_mult(
                c1,
                c2,
                return_selector_count=True,
            )
        )

        decoded = (
            he.decrypt(
                c_mul
            )
        )

        expected = (
            he.plaintext_multiply(
                mu1,
                mu2,
            )
        )

        selector_counts.append(
            selected
        )

        if bits_equal(
            decoded,
            expected,
        ):
            multiplication_successes += 1

    print(
        "Exact fused multiplication       : "
        f"{multiplication_successes}/"
        f"{multiplication_trials}"
    )

    print(
        "Average selected fused rows       : "
        f"{np.mean(selector_counts):.3f}"
    )

    print(
        "Min/max selected fused rows       : "
        f"{min(selector_counts)}/"
        f"{max(selector_counts)}"
    )

    if (
        multiplication_successes
        !=
        multiplication_trials
    ):
        raise AssertionError(
            "Exact fused multiplication failed."
        )

    # ==========================================================
    # 8. Test a nonzero admissible encryption error when
    #    a weight-1 admissible coordinate exists.
    # ==========================================================

    singleton_coordinates = (
        np.flatnonzero(
            np.all(
                he.Lambda_msg
                ==
                0,
                axis=1,
            )
        )
    )

    print()
    print(
        "[Admissible encryption-noise test]"
    )

    print(
        "Weight-1 admissible coordinates  : "
        f"{len(singleton_coordinates)}"
    )

    if (
        len(
            singleton_coordinates
        )
        >
        0
    ):
        old_tau_enc = (
            he.tau_enc
        )

        he.tau_enc = 1

        admissible_successes = 0

        for _ in range(
            5
        ):
            mu = he.rng.integers(
                0,
                2,
                size=ell,
                dtype=np.uint8,
            )

            (
                ciphertext,
                metadata,
            ) = (
                he.encrypt(
                    mu,
                    return_metadata=True,
                )
            )

            neutral = (
                gf2_matmul(
                    metadata[
                        "error"
                    ],
                    he.Lambda_msg,
                )
            )

            decoded = (
                he.decrypt(
                    ciphertext
                )
            )

            if (
                metadata[
                    "error_weight"
                ]
                ==
                1
                and
                not np.any(
                    neutral
                )
                and
                bits_equal(
                    decoded,
                    mu,
                )
            ):
                admissible_successes += 1

        he.tau_enc = (
            old_tau_enc
        )

        print(
            "Nonzero admissible encryptions   : "
            f"{admissible_successes}/5"
        )

        if admissible_successes != 5:
            raise AssertionError(
                "Admissible-noise encryption failed."
            )

    else:
        print(
            "Nonzero weight-1 test            : SKIPPED"
        )

    # ==========================================================
    # 9. Noisy fused-key diagnostic, tau_F = 1
    #
    # This is intentionally diagnostic. Dense selector
    # accumulation can exceed the decoder radius.
    # ==========================================================

    print()
    print(
        "[Noisy fused-key diagnostic: tau_F=1]"
    )

    noisy_evk = (
        he.generate_evaluation_key(
            tau_F=1,
            batch_size=64,
            max_pairs=200000,
            progress=False,
            set_default=False,
        )
    )

    noisy_verification = (
        he.verify_evaluation_key(
            noisy_evk,
            samples=min(
                32,
                noisy_evk.pair_count,
            ),
        )
    )

    noisy_rows_ok = (
        noisy_verification[
            "all_rows_have_expected_noise_weight"
        ]
    )

    print(
        "Each sampled row has wt(z)=1     : "
        f"{'PASS' if noisy_rows_ok else 'FAIL'}"
    )

    if not noisy_rows_ok:
        raise AssertionError(
            "Noisy fused rows are malformed."
        )

    mu1 = he.rng.integers(
        0,
        2,
        size=ell,
        dtype=np.uint8,
    )

    mu2 = he.rng.integers(
        0,
        2,
        size=ell,
        dtype=np.uint8,
    )

    c1 = he.encrypt(
        mu1
    )

    c2 = he.encrypt(
        mu2
    )

    noisy_diag = (
        he.multiplication_diagnostics(
            c1,
            c2,
            evk=noisy_evk,
        )
    )

    print(
        "Selected Y rows                  : "
        f"{noisy_diag['selected_rows']}"
    )

    print(
        "Actual fused output noise weight : "
        f"{noisy_diag['fused_noise_weight']}"
    )

    print(
        "Goppa decoding radius            : "
        f"{noisy_diag['decoder_radius']}"
    )

    print(
        "Noise within decoding radius     : "
        f"{noisy_diag['within_decoder_radius']}"
    )

    print(
        "Noisy multiplication decrypted   : "
        f"{noisy_diag['decode_success']}"
    )

    print()
    print(
        "PARAMETER SET PASSED."
    )


def main():
    run_parameter_set(
        m=4,
        n=15,
        t=2,
        ell=2,
        seed=301,
        multiplication_trials=20,
    )

    run_parameter_set(
        m=5,
        n=31,
        t=2,
        ell=2,
        seed=302,
        multiplication_trials=10,
    )

    print()
    print("=" * 88)

    print(
        "PHASE 3A-3D FUSED HE "
        "REGRESSION PASSED."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()