from pathlib import Path

import numpy as np
from ngio import (
    Image,
    Label,
    OmeZarrContainer,
    create_empty_ome_zarr,
    open_ome_zarr_container,
)
from ngio.utils import download_ome_zarr_dataset

# We download a sample Ome-Zarr and use it as a source to create smaller test datasets

ROOT_PATH: Path = Path("./data")

DOWNLOAD_DIR_PATH: Path = ROOT_PATH / "downloaded_dataset"

source_hcs_path = download_ome_zarr_dataset(
    "CardiomyocyteSmall",
    download_dir=ROOT_PATH / "downloaded_dataset",
)
source_image_path = source_hcs_path / "B/03/0"

source_ome_zarr: OmeZarrContainer = open_ome_zarr_container(source_image_path)
print(f"{source_ome_zarr=}")
source_image: Image = source_ome_zarr.get_image("2")
print(f"{source_image=}")


# prepare target container
source_array: np.ndarray = source_image.get_array()

target_container = create_empty_ome_zarr(
    ROOT_PATH / "test_data.zarr/A/01/0",
    shape=source_array.shape,
    xy_pixelsize=source_image.pixel_size.x,
    levels=3,
    overwrite=True,
)

# prepare target image
target_image: Image = target_container.get_image()
target_image.set_array(source_array)
target_image.consolidate()

# prepare label
source_label: Label = source_ome_zarr.get_label(
    "nuclei",
    pixel_size=source_image.pixel_size,
)
source_label_array: np.ndarray = source_label.get_array()
target_label: Label = target_container.derive_label(
    name="nuclei",
    overwrite=True,
    ref_image=target_image,
)
target_label.set_array(source_label_array)
target_label.consolidate()

# adding masking roi table for the label
target_container.add_table(
    name="nuclei_ROI_table",
    table=target_label.build_masking_roi_table(),
    # backend="parquet",
    overwrite=True,
)

print(f"{target_container=}")
