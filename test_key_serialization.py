"""
Phase 2D key serialization regression.

The private key stores

    J = right_inverse_pivots
    C = right_inverse_core

with

    G[:,J] C = I_k.

The old dense n x k R must not appear in the serialized key.
"""

import importlib.util
from pathlib import Path
import tempfile

import numpy as np

from mceliece.mathutils import GF2Matrix


M = 6
N = 63
T = 3

TRIALS = 10
RANDOM_SEED = 20260816

EXPECTED_KEY_FORMAT_VERSION = 4


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parent
)

CLI_FILE = (
    PROJECT_ROOT
    / "mceliece.py"
)


def load_cli_module():

    spec = (
        importlib.util.spec_from_file_location(
            "mceliece_cli",
            CLI_FILE,
        )
    )

    if spec is None:
        raise RuntimeError(
            "Unable to load mceliece.py."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    if spec.loader is None:
        raise RuntimeError(
            "Module loader unavailable."
        )

    spec.loader.exec_module(
        module
    )

    return module


def gf2_numpy(matrix):

    return np.array(
        [
            int(x.n)
            for x in matrix.arr.flat
        ],
        dtype=np.uint8,
    ).reshape(
        matrix.arr.shape
    )


def matrix_equal(
    A,
    B,
):

    return np.array_equal(
        gf2_numpy(
            A
        ),
        gf2_numpy(
            B
        ),
    )


def identity_gf2(k):

    return GF2Matrix.from_list(
        np.eye(
            k,
            dtype=np.uint8,
        )
    )


def main():

    np.random.seed(
        RANDOM_SEED
    )

    cli = load_cli_module()

    print()
    print("=" * 78)

    print(
        "PHASE 2D KEY-SERIALIZATION REGRESSION"
    )

    print("=" * 78)

    with tempfile.TemporaryDirectory(
        prefix="mceliece_phase2d_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        private_key_file = (
            temp_dir
            / "private_key.npz"
        )

        public_key_file = (
            temp_dir
            / "public_key.npz"
        )

        # --------------------------------------------------------
        # Generate.
        # --------------------------------------------------------

        print()
        print(
            "[1] Generating keys..."
        )

        generated = cli.generate(
            M,
            N,
            T,
            private_key_file,
            public_key_file,
        )

        k = int(
            generated.k
        )

        J = (
            generated
            .right_inverse_pivots
        )

        C = (
            generated
            .right_inverse_core
        )

        print(
            "  k          =",
            k,
        )

        print(
            "  J shape    =",
            J.shape,
        )

        print(
            "  C shape    =",
            C.arr.shape,
        )

        assert J.shape == (
            k,
        )

        assert C.arr.shape == (
            k,
            k,
        )

        assert not hasattr(
            generated,
            "R",
        )

        # --------------------------------------------------------
        # Inspect private key.
        # --------------------------------------------------------

        print()
        print(
            "[2] Inspecting private key..."
        )

        with np.load(
            private_key_file,
            allow_pickle=False,
        ) as data:

            fields = set(
                data.files
            )

            version = int(
                np.asarray(
                    data[
                        "format_version"
                    ]
                ).item()
            )

            G_array = np.asarray(
                data["G"],
                dtype=np.uint8,
            )

            pivots = np.asarray(
                data[
                    "right_inverse_pivots"
                ],
                dtype=np.int64,
            )

            core_array = np.asarray(
                data[
                    "right_inverse_core"
                ],
                dtype=np.uint8,
            )

        print(
            "  format version             =",
            version,
        )

        print(
            "  G shape                    =",
            G_array.shape,
        )

        print(
            "  right_inverse_pivots shape =",
            pivots.shape,
        )

        print(
            "  right_inverse_core shape   =",
            core_array.shape,
        )

        print(
            "  dense R field present      =",
            "R" in fields,
        )

        assert version == (
            EXPECTED_KEY_FORMAT_VERSION
        )

        assert G_array.shape == (
            k,
            N,
        )

        assert pivots.shape == (
            k,
        )

        assert core_array.shape == (
            k,
            k,
        )

        assert "R" not in fields

        assert len(
            np.unique(
                pivots
            )
        ) == k

        # --------------------------------------------------------
        # Direct factorization check.
        # --------------------------------------------------------

        G = GF2Matrix.from_list(
            G_array
        )

        core = GF2Matrix.from_list(
            core_array
        )

        B = GF2Matrix(
            G.arr[
                :,
                pivots
            ]
        )

        factorization_ok = matrix_equal(
            B * core,
            identity_gf2(
                k
            ),
        )

        print(
            "  G[:,J] C = I_k             =",
            factorization_ok,
        )

        assert factorization_ok

        # --------------------------------------------------------
        # Reload.
        # --------------------------------------------------------

        print()
        print(
            "[3] Reloading private/public keys..."
        )

        private_mc = (
            cli.load_private_key(
                private_key_file
            )
        )

        public_mc = (
            cli.load_public_key(
                public_key_file
            )
        )

        assert (
            private_mc
            .right_inverse_pivots
            .shape
            ==
            (
                k,
            )
        )

        assert (
            private_mc
            .right_inverse_core
            .arr
            .shape
            ==
            (
                k,
                k,
            )
        )

        assert np.array_equal(
            generated.right_inverse_pivots,
            private_mc.right_inverse_pivots,
        )

        assert matrix_equal(
            generated.right_inverse_core,
            private_mc.right_inverse_core,
        )

        print(
            "  factorized right inverse round trip : PASS"
        )

        # --------------------------------------------------------
        # Public generator.
        # --------------------------------------------------------

        SG = (
            private_mc.S
            * private_mc.G
        )

        reconstructed_Gp = GF2Matrix(
            SG.arr[
                :,
                private_mc.P
            ]
        )

        assert matrix_equal(
            reconstructed_Gp,
            public_mc.Gp,
        )

        print(
            "  public generator round trip         : PASS"
        )

        # --------------------------------------------------------
        # Clean extraction after reload.
        # --------------------------------------------------------

        print()
        print(
            "[4] Clean extraction after reload..."
        )

        for trial in range(
            1,
            11,
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
                * private_mc.G
            )

            recovered = (
                private_mc
                ._extract_information_vector(
                    codeword
                )
            )

            ok = np.array_equal(
                recovered
                .to_numpy()
                .astype(
                    np.uint8
                ),
                a_numpy,
            )

            print(
                f"  clean trial {trial:02d}: "
                f"{ok}"
            )

            assert ok

        # --------------------------------------------------------
        # Prevent factorization reconstruction.
        # --------------------------------------------------------

        print()
        print(
            "[5] Disabling factorization reconstruction..."
        )

        original_constructor = (
            cli.McElieceCipher
            ._construct_right_inverse_factorization
        )

        def forbidden_constructor(
            self,
        ):
            raise AssertionError(
                "Loaded Phase 2D key attempted "
                "to reconstruct its factorization."
            )

        (
            cli.McElieceCipher
            ._construct_right_inverse_factorization
        ) = forbidden_constructor

        # --------------------------------------------------------
        # Disk encryption/decryption.
        # --------------------------------------------------------

        successes = 0

        try:

            print()
            print(
                f"[6] Running {TRIALS} disk-key trials..."
            )

            for trial in range(
                1,
                TRIALS + 1,
            ):

                message = np.random.randint(
                    0,
                    2,
                    size=k,
                    dtype=np.uint8,
                )

                ciphertext = cli.encrypt(
                    public_key_file,
                    message,
                    block=False,
                )

                decoded = cli.decrypt(
                    private_key_file,
                    ciphertext,
                    block=False,
                )

                success = np.array_equal(
                    message,
                    decoded,
                )

                if success:
                    successes += 1

                print(
                    f"  trial {trial:02d}: "
                    f"{success}"
                )

                assert success

        finally:

            (
                cli.McElieceCipher
                ._construct_right_inverse_factorization
            ) = original_constructor

        # --------------------------------------------------------
        # Final summary.
        # --------------------------------------------------------

        print()
        print("=" * 78)

        print(
            "PHASE 2D SERIALIZATION RESULT"
        )

        print("=" * 78)

        print(
            "Key format version 4             : PASS"
        )

        print(
            "allow_pickle=False               : PASS"
        )

        print(
            "Dense R absent                   : PASS"
        )

        print(
            "Pivot vector persisted           : PASS"
        )

        print(
            "Right-inverse core persisted     : PASS"
        )

        print(
            "G[:,J] C = I_k                  : PASS"
        )

        print(
            "(aG)[J] C = a after reload      : PASS"
        )

        print(
            "No reconstruction on decrypt    : PASS"
        )

        print(
            "Public generator round trip      : PASS"
        )

        print(
            "Patterson after reload           : PASS"
        )

        print(
            "Disk encryption/decryption       : "
            f"{successes}/{TRIALS} PASS"
        )

        print()
        print(
            "PHASE 2D KEY SERIALIZATION "
            "WORKS CORRECTLY."
        )


if __name__ == "__main__":
    main()