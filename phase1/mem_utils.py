"""
mem_utils.py
Memory-optimization helpers shared by every phase of the pipeline.

IMPORTANT DESIGN NOTE (flagged for your methods section):
Your original spec said "always downcast int64 -> int8". Applied blindly,
that corrupts real financial data: int8 only holds -128..127, but a single
column like `credit_amount` or `RevolvingUtilization` routinely holds
values in the thousands. Silently forcing those into int8 would wrap
around (e.g. 9055 -> some garbage value < 128) and destroy the dataset
without raising an error.

Instead, `downcast_dataframe()` below performs *safe* downcasting:
  - float64 -> float32                         (always; SHAP/LIME/trees
                                                  tolerate this precision
                                                  loss trivially)
  - int64   -> smallest int type that actually
               fits the column's observed min/max
               (int8 for binary flags / small ordinal codes,
                int16/int32 for magnitude columns)
This still gets you the RAM savings (one-hot dummy columns *do* become
int8, which is most of your column count after encoding) without any
silent data corruption.
"""

import gc
import numpy as np
import pandas as pd


def downcast_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast float64->float32 and int64->smallest safe int dtype.

    Returns a new DataFrame (does not mutate in place) so callers can
    `del` the original and gc.collect() explicitly, per project convention.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    df = df.copy()

    for col in df.columns:
        col_dtype = df[col].dtype

        if col_dtype == np.float64:
            df[col] = df[col].astype(np.float32)

        elif col_dtype == np.int64:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= -128 and c_max <= 127:
                df[col] = df[col].astype(np.int8)
            elif c_min >= -32768 and c_max <= 32767:
                df[col] = df[col].astype(np.int16)
            elif c_min >= -2147483648 and c_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
            # else: leave as int64 (rare — would need a value beyond ~2.1B)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(f"  [downcast] {start_mem:6.2f} MB -> {end_mem:6.2f} MB  "
              f"({pct:5.1f}% reduction)")

    gc.collect()
    return df


def report_memory(df: pd.DataFrame, name: str = "df") -> None:
    """Quick memory usage printout, used for sanity checks between phases."""
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"  [memory] {name}: shape={df.shape}, {mem_mb:.2f} MB, "
          f"dtypes={df.dtypes.value_counts().to_dict()}")
