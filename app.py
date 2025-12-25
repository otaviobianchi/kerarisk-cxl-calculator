import re
import skops.io as sio
from pathlib import Path


def safe_skops_load(path: Path, debug=False):
    """
    CVE-safe loader for skops >= 0.10
    - trusted MUST be list[str]
    - NEVER uses trusted=True
    - NEVER calls get_untrusted_types(path)
    """

    # Minimal baseline (safe + common sklearn objects)
    trusted = [
        "sklearn.pipeline.Pipeline",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.impute._base.SimpleImputer",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.linear_model._coordinate_descent.ElasticNet",
        "numpy.ndarray",
        "numpy.dtype",
    ]

    try:
        return sio.load(path, trusted=trusted), trusted
    except Exception as e:
        msg = str(e)

        if "Untrusted types found in the file" not in msg:
            raise RuntimeError(f"Failed to load {path.name}: {e}")

        # 🔐 Extract untrusted types from exception message
        # skops prints them as:
        #  - module.ClassName
        extra = re.findall(r"^\s*-\s*(.+?)\s*$", msg, flags=re.MULTILINE)

        if not extra:
            raise RuntimeError(
                "skops blocked model loading but no untrusted types could be extracted.\n"
                "This usually indicates a corrupted .skops file."
            )

        trusted_final = sorted(set(trusted + extra))

        if debug:
            print(f"[skops] Auto-trusting {len(extra)} extra types from {path.name}:")
            for t in extra:
                print("  -", t)

        return sio.load(path, trusted=trusted_final), trusted_final
