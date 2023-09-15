import numpy as np
from pathlib import Path

data_file = Path(
    "/local/home/xkan/ABCD/ABCD/abcd_rest-timeseires-HCP2016-1024.npy")

ts_data = np.load(
    data_file, allow_pickle=True)

a, b, _ = ts_data.shape

# np.mean(arr.reshape(-1, 3), axis=1)

for leng in (64, 128, 256, 512, 1024):

    data = np.mean(ts_data.reshape(a, b, -1, 1024//leng),
                   axis=-1, keepdims=False)
    all_sample = []
    for d in data:

        m = np.corrcoef(d)
        all_sample.append(m)
        if np.isnan(m).any():
            print(leng, "nan")

    m = np.array(all_sample)
    print(m.shape)
    np.save(
        data_file.parent / f"abcd_rest-pearson-HCP2016-{leng}.npy", m)
