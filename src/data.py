import urllib
import shutil
from tempfile import TemporaryDirectory
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, Tuple

import pandas as pd

class _DownloadProgressBar(tqdm):
    def update_to(
        self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None
    ) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_url(url: str, output_path: Path) -> None:
    with _DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to
        )  # nosec

# %% ../../nbs/Experiments.ipynb 8
def _get_data_path(data_path: Optional[Union[Path, str]] = None) -> Path:
    if data_path is None:
        data_path = "./data"
    return Path(data_path)


def _download_data(
    dataset_name: str,
    data_path: Optional[Union[Path, str]] = "data",
    force_download: bool = False,
) -> None:
    data_path = _get_data_path(data_path)
    data_path.mkdir(exist_ok=True, parents=True)

    for prefix in ["train", "test"]:
        filename = f"{prefix}_{dataset_name}.csv"
        if not (data_path / filename).exists() or force_download:
            with TemporaryDirectory() as d:
                _download_url(
                    f"https://zenodo.org/record/7968969/files/{filename}",
                    Path(d) / filename,
                )
                shutil.copyfile(Path(d) / filename, data_path / filename)
        else:
            print(f"Upload skipped, file {(data_path / filename).resolve()} exists.")

# %% ../../nbs/Experiments.ipynb 10
def _sanitize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    columns = {c: c.replace(" ", "_") for c in df}
    df = df.rename(columns=columns)
    return df

# %% ../../nbs/Experiments.ipynb 12
def get_train_n_test_data(
    dataset_name: str,
    *,
    data_path: Optional[Union[Path, str]] = "./data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download data

    Args:
        dataset_name: name of the dataset, one of "auto", "heart", compas", "blog", "loan"
        data_path: root directory where to download data to
    """
    data_path = _get_data_path(data_path)
    _download_data(dataset_name=dataset_name, data_path=data_path)

    dfx = [
        pd.read_csv(data_path / f"{prefix}_{dataset_name}.csv")
        for prefix in ["train", "test"]
    ]
    dfx = [_sanitize_col_names(df) for df in dfx]
    return dfx[0], dfx[1]