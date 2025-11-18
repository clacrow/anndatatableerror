from ngio import open_ome_zarr_container

ome_zarr = open_ome_zarr_container("data/test_data.zarr/A/01/0")

table = ome_zarr.get_generic_roi_table("nuclei_ROI_table")
print("table as dataframe:")
print(f"{table.dataframe=}")
print("table rois:")
print(f"{table.rois()=}")
