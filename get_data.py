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
from ngio.ome_zarr_meta.ngio_specs import Channel
from ngio.tables import RoiTable
from ngio.utils import (
    download_ome_zarr_dataset,
)

# We download a sample Ome-Zarr and use it as a source to create smaller test datasets

ROOT_PATH: Path = Path("./data")

TARGET_DATASET_PATH: Path = ROOT_PATH / "test_data.zarr"
TARGET_DATASET_NAME: str = "Test Dataset"
TARGET_DATASET_LEVELS: int = 3


DOWNLOAD_DIR_PATH: Path = ROOT_PATH / "downloaded_source_dataset"
DOWNLOAD_PATH: Path = (
    DOWNLOAD_DIR_PATH / "20200812-CardiomyocyteDifferentiation14-Cycle1-small.zarr"
)
DOWNLOAD_DATASET_NAME: str = "CardiomyocyteSmall"
DOWNLOAD_IMAGE_PATH: str = "B/03/0"
DOWNLOAD_IMAGE_LEVEL: str = "2"  # we take level 2 for smaller size

print("Downloading source dataset...")
source_hcs_path = download_ome_zarr_dataset(
    "CardiomyocyteSmall",
    download_dir=ROOT_PATH / "downloaded_dataset",
)
source_image_path = source_hcs_path / DOWNLOAD_IMAGE_PATH

source_ome_zarr: OmeZarrContainer = open_ome_zarr_container(source_image_path)
print(f"{source_ome_zarr=}")
source_image: Image = source_ome_zarr.get_image(DOWNLOAD_IMAGE_LEVEL)
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
source_channels: list[Channel] = source_image.channels_meta.channels

target_container = create_empty_ome_zarr(
    TARGET_DATASET_PATH / "A/01/0",
    shape=source_array.shape,
    xy_pixelsize=source_image.pixel_size.x,
    z_spacing=source_image.pixel_size.z,
    time_spacing=source_image.pixel_size.t,
    levels=TARGET_DATASET_LEVELS,
    space_unit=source_image.space_unit,
    time_unit=source_image.time_unit,
    axes_names=source_image.dimensions.axes,
    dtype=source_image.dtype,
    channel_labels=source_image.channel_labels,
    overwrite=True,
)
target_container._images_container.set_channel_meta(  # noqa: SLF001 cannot assign otherwise
    labels=source_image.channel_labels,
    colors=[ch.channel_visualisation.color for ch in source_channels],
    active=[ch.channel_visualisation.active for ch in source_channels],
    start=[ch.channel_visualisation.start for ch in source_channels],
    end=[ch.channel_visualisation.end for ch in source_channels],
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

more_rois: list[Roi] = [
    Roi(
        name=n,
        x=image_roi.x,
        y=image_roi.y,
        z=image_roi.z,
        x_length=image_roi.x_length / (i + 1),
        y_length=image_roi.y_length / (i + 1),
        z_length=image_roi.z_length,
    )
    for i, n in enumerate(["ROI_1", "ROI_2", "ROI_3", "ROI_4"])
]


target_container.add_table(
    name="FOV_ROI_table",
    table=RoiTable(more_rois),
    backend="parquet",
    overwrite=True,
)

print(target_container)
print(f"{target_image.is_2d=}")
print(f"{target_image.is_3d=}")
