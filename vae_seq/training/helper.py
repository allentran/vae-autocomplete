
import numpy as np
from scipy.interpolate import interpolate


def long_df_to_array(df, groupby_cols, meta_cols, val_col, idx_col, min_idx=1, min_length=5, length=10):

    df = df.sort_values(by=groupby_cols)

    sequences = []
    meta_data = []
    og_length = []

    for name, group in df.groupby(groupby_cols):

        if group.shape[0] < min_length or group[idx_col].min() > min_idx:
            continue
        interpolater = interpolate.interp1d(group[idx_col], group[val_col], bounds_error=True)
        x_grid = np.linspace(1, group[idx_col].max(), length)

        og_length.append(len(group))
        sequences.append(interpolater(x_grid))
        meta_data.append(group[meta_cols].values[0, :])

    return np.vstack(sequences), np.vstack(meta_data), og_length


