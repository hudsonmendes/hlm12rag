# Python Built-in Modules
import pathlib
import shutil
import zipfile

# Third-Party Libraries
import kaggle

DIR_TMP = pathlib.Path("/tmp/kaggle")


def etl_data_from_kaggle(dataset: str, dst: pathlib) -> None:
    """
    Functional entrypoint for ETL pipeline.

    Arg
    """
    Etl(dataset=dataset).process_into(dst=dst)


class Etl:
    dir_raw: pathlib.Path
    dir_transformed: pathlib.Path

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.dir_base = DIR_TMP / dataset

    def process_into(self, dst: pathlib.Path) -> None:
        filepath_zip = self._extract(dataset=self.dataset)
        dirpath_transformed = self._transform(filepath_zip=filepath_zip)
        self._load(src=dirpath_transformed, dst=dst)

    def _extract(self, dataset: str) -> pathlib.Path:
        dir_raw = self.dir_base / "raw"
        kaggle.api.dataset_download_cli(dataset, path=dir_raw)
        return pathlib.Path(next(dir_raw.glob("*.zip")))

    def _transform(self, filepath_zip: pathlib.Path) -> pathlib.Path:
        dir_transformed = self.dir_base / "transformed"
        dir_transformed.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(filepath_zip) as zf:
            for zfi in zf.filelist:
                if not zfi.is_dir() and zfi.filename.endswith(".clean"):
                    filename = ".".join(zfi.filename.split("/")[-1].split(".")[:-1])
                    filepath_src = zf.extract(zfi)
                    filepath_dst = dir_transformed / filename
                    shutil.move(filepath_src, filepath_dst)
        return dir_transformed

    def _load(self, src: pathlib.Path, dst: pathlib.Path) -> None:
        shutil.copytree(src, dst, dirs_exist_ok=True)
