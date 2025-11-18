from pathlib import Path

import numpy as np
from ngio import (
    Image,
    ImageInWellPath,
    Label,
    OmeZarrContainer,
    OmeZarrPlate,
    Roi,
    create_empty_ome_zarr,
    create_empty_plate,
    open_ome_zarr_container,
)
from ngio.utils import download_ome_zarr_dataset

# We download a sample Ome-Zarr and use it as a source to create smaller test datasets

ROOT_PATH: Path = Path("./data")

TARGET_DATASET_PATH: Path = ROOT_PATH / "test_data.zarr"
TARGET_DATASET_NAME: str = "Test Dataset"


DOWNLOAD_DIR_PATH: Path = ROOT_PATH / "downloaded_dataset"
DOWNLOAD_PATH: Path = (
    DOWNLOAD_DIR_PATH / "20200812-CardiomyocyteDifferentiation14-Cycle1-small.zarr"
)

print("Downloading source dataset...")
source_hcs_path = download_ome_zarr_dataset(
    "CardiomyocyteSmall",
    download_dir=ROOT_PATH / "downloaded_dataset",
)
source_image_path = source_hcs_path / "B/03/0"

source_ome_zarr: OmeZarrContainer = open_ome_zarr_container(source_image_path)
print(f"{source_ome_zarr=}")
source_image: Image = source_ome_zarr.get_image("2")
print(f"{source_image=}")

image_roi: Roi = source_ome_zarr.build_image_roi_table().rois()[0]

target_plate_wells = [ImageInWellPath(row="A", column="01", path="0")]
test_3d_plate: OmeZarrPlate = create_empty_plate(
    store=TARGET_DATASET_PATH,
    name=TARGET_DATASET_NAME,
    images=target_plate_wells,
    overwrite=True,
)

# prepare target container
source_array: np.ndarray = source_image.get_roi(image_roi)

target_container = create_empty_ome_zarr(
    TARGET_DATASET_PATH / "A/01/0",
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
source_label_array: np.ndarray = source_label.get_roi(image_roi)
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
