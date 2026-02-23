import pvdeg
import pvlib
import pandas as pd
import numpy as np
import xarray as xr
from dask.distributed import LocalCluster, Client
from dask_jobqueue import SLURMCluster

PVmodule = {
    "bifaciality": 0.0,
    "thermal_a": -3.47,
    "thermal_b": -0.0594,
    "thermal_dT": 3,
    "pdc0": 100,
    "gamma_pdc": -0.002,
}

trackingSystem = {
    "gcr": 0.33,
    "axis_tilt": 0,
    "axis_azimuth": 180,
    "backtracking": True,
    "max_angle": 50,
    "cross_axis_tilt": 0,
    "module_height": 1.5,
    "pitch": 5.7,
    "albedo": 0.2,
}

def offset_to_gmt(offset):
    sign = "+" if offset <= 0 else "-"
    return f"Etc/GMT{sign}{int(abs(offset))}"

def NSRDB_localize(weather_df, meta):
    zone_str = offset_to_gmt(meta['tz'])
    local_weather = weather_df.tz_localize("gmt").tz_convert(zone_str)
    return local_weather 

# rename diffuse_stow to mainfunction
@pvdeg.decorators.geospatial_quick_shape("numeric", ["DNI_Wperm2", "POA_Wperm2"])  

def mainfunction(weather_df: pd.DataFrame, meta: dict, time_index=205) -> pd.DataFrame:
    weather_df = NSRDB_localize(weather_df, meta)

    time_index = weather_df.index

    site_loc = pvlib.location.Location(meta["latitude"], meta["longitude"])
    dni_clear = site_loc.get_clearsky(time_index)["dni"]

    dni_time_index = dni_clear.iloc[time_index]
    sp = site_loc.get_solarposition(time_index)

    rotation = pvlib.tracking.singleaxis(
        apparent_zenith=sp["apparent_zenith"],
        apparent_azimuth=sp["azimuth"],
        axis_tilt=trackingSystem["axis_tilt"],
        axis_azimuth=trackingSystem["axis_azimuth"],
        backtrack=trackingSystem["backtracking"],
        gcr=trackingSystem["gcr"],
        max_angle=trackingSystem["max_angle"],
        cross_axis_tilt=trackingSystem["cross_axis_tilt"],
    )

    tilt_TT = rotation["surface_tilt"].fillna(0)

    irrad_TT = pvlib.bifacial.infinite_sheds.get_irradiance(
        surface_tilt=tilt_TT,
        surface_azimuth=rotation["surface_azimuth"],
        solar_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        gcr=trackingSystem["gcr"],
        height=trackingSystem["module_height"],
        pitch=trackingSystem["pitch"],
        ghi=weather_df["ghi"],
        dhi=weather_df["dhi"],
        dni=weather_df["dni"],
        albedo=trackingSystem["albedo"],
        bifaciality=PVmodule["bifaciality"],
    )

    POA_time_index = irrad_TT.iloc[time_index]

    df_result = pd.DataFrame(
        {
            "DNI_Wperm2": dni_time_index,,
            "POA_Wperm2": POA_time_index,
        },
        index=[
            0,
        ],
    )

    return df_result

cluster = SLURMCluster(
    queue='shared',
    account="pvsoiling",
    cores=2,
    memory="30 GB",
    processes=True,
    log_directory='/scratch/sayala/dev/logs',
    walltime="02:00:00",
)
cluster.scale(128)
client = Client(cluster)
print(client.dashboard_link)


geo_weather = xr.open_zarr("/projects/pvsoiling/pvdeg/data/GOES_CONUS_5min/CONUS_5min.zarr")

sub_tmy_meta = pd.read_csv("/projects/pvsoiling/pvdeg/data/GOES_CONUS_5min/meta.csv", index_col=0)

geo_weather = geo_weather.chunk({"gid":5})


combined_gids = geo_weather.gid.values

# make sure this ends with a / so its interpreted as a directory
target_dir = "/projects/pvsoiling/pvdeg/analysis/frontIrradianceStudy/"
step_size = 64*5 # chunks of 5 and 128 workers
for i in range(0, len(combined_gids), step_size):

    front = i
    back = min(i + step_size, len(combined_gids) - 1) # ensure we dont go out of bounds

    slice_weather = geo_weather.isel(gid=slice(front, back))
    slice_meta = sub_tmy_meta.iloc[front : back]

    partial_res = pvdeg.geospatial.analysis(
        weather_ds = slice_weather,
        meta_df = slice_meta,
        func = mainfunction
    )

    partial_res.to_netcdf(f"{target_dir}-{i}-{i+step_size-1}.nc")
    print("ended", i)
