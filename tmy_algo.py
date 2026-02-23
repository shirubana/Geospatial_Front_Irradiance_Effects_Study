import pvdeg
import pvlib
import pandas as pd
import numpy as np
from dask.distributed import LocalCluster, Client
from dask_jobqueue import SLURMCluster

def offset_to_gmt(offset):
    sign = "+" if offset <= 0 else "-"
    return f"Etc/GMT{sign}{abs(offset)}"

def NSRDB_localize(weather_df, meta):
    zone_str = offset_to_gmt(meta['tz'])
    local_weather = weather_df.tz_localize("gmt").tz_convert(zone_str)
    return local_weather 

PVmodule = {
    "bifaciality": 0.70,
    "thermal_a": -3.47,
    "thermal_b": -0.0594,
    "thermal_dT": 3,
    "pdc0": 100,
    "gamma_pdc": -0.002,
}
trackingSystem = {
    "gcr": 0.4,
    "axis_tilt": 0,
    "axis_azimuth": 180,
    "backtracking": True,
    "max_angle": 55,
    "cross_axis_tilt": 0,
    "module_height": 1.5,
    "pitch": 5.7,
    "albedo": 0.2,
}


@pvdeg.decorators.geospatial_quick_shape("numeric", ["delta_mov (%)", "delta_en (%)", "energy_TT (Wh)"])  
def diffuse_stow(weather_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    weather_df = NSRDB_localize(weather_df, meta)

    time_index = weather_df.index

    site_loc = pvlib.location.Location(meta["latitude"], meta["longitude"])
    dni_clear = site_loc.get_clearsky(time_index)["dni"]
    sp = site_loc.get_solarposition(time_index)

    weather_df["Cdir"] = weather_df["dni"] / dni_clear

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

    temp_cell_TT = pvlib.temperature.sapm_cell(
        irrad_TT["poa_global"],
        temp_air=weather_df["temp_air"],
        wind_speed=weather_df["wind_speed"],
        a=PVmodule["thermal_a"],
        b=PVmodule["thermal_b"],
        deltaT=PVmodule["thermal_dT"],
    )

    pdc_TT = pvlib.pvsystem.pvwatts_dc(
        irrad_TT["poa_global"], temp_cell_TT, PVmodule["pdc0"], PVmodule["gamma_pdc"]
    ).fillna(0)

    tilt_00 = pd.Series(0, index=tilt_TT.index)
    irrad_00 = pvlib.bifacial.infinite_sheds.get_irradiance(
        surface_tilt=tilt_00,
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

    temp_cell_00 = pvlib.temperature.sapm_cell(
        irrad_00["poa_global"],
        temp_air=weather_df["temp_air"],
        wind_speed=weather_df["wind_speed"],
        a=PVmodule["thermal_a"],
        b=PVmodule["thermal_b"],
        deltaT=PVmodule["thermal_dT"],
    )
    pdc_00 = pvlib.pvsystem.pvwatts_dc(
        irrad_00["poa_global"], temp_cell_00, PVmodule["pdc0"], PVmodule["gamma_pdc"]
    ).fillna(0)


    GF_mask = weather_df["Cdir"] <= 0.1
    pdc_GF = pdc_00.where(GF_mask, pdc_TT)
    tilt_GF = tilt_00.where(GF_mask, tilt_TT)

    mov_GF = tilt_GF.diff()
    mov_TT = tilt_TT.diff()
    mov_GF = mov_GF.dropna()
    mov_TT = mov_TT.dropna()
    mov_GF = np.abs(mov_GF)
    mov_TT = np.abs(mov_TT)
    mov_GF_cum = sum(mov_GF)
    mov_TT_cum = sum(mov_TT)

    energy_GF = sum(pdc_GF)  # /(60/30)
    energy_TT = sum(pdc_TT)  # /(60/30)

    # # For debugging only -- remove afterwards
    # weather_df["dni_clear"] = dni_clear

    # weather_df['tilt_00'] = tilt_00
    # weather_df['pdc_00'] = pdc_00

    # weather_df['mov_TT'] = mov_TT
    # weather_df['pdc_TT'] = pdc_TT

    # weather_df['GF_mask'] = GF_mask
    # weather_df['tilt_GF'] = tilt_GF
    # weather_df['pdc_GF'] = pdc_GF
    # weather_df['mov_GF'] = mov_GF

    # weather_df.to_csv("internal_debug.csv")


    # not sure if it is appropriate to set these to zeros instead of dividing by zero
    try:
        delta_mov_perc = 100 * (mov_GF_cum - mov_TT_cum) / mov_TT_cum
    except:
        delta_mov_perc = 0

    try:
        delta_en_perc = 100 * (energy_GF - energy_TT) / energy_TT
    except:
        delta_en_perc = 0

    df_result = pd.DataFrame(
        {
            "delta_mov (%)": delta_mov_perc,
            "delta_en (%)": delta_en_perc,
            "energy_TT (Wh)": energy_TT,
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
    log_directory='/scratch/tford/dev/logs',
    walltime="02:00:00",
)
cluster.scale(64)
client = Client(cluster)
print(client.dashboard_link)

# geo weather as 8208 chunk entries in groups of 10 so 821 chunks

weather_db = "NSRDB"
weather_arg = {
    "satellite": "Americas",
    "names": "TMY",
    "NREL_HPC": True,
    "attributes": [
            "air_temperature",
            "wind_speed",
            "dhi",
            "ghi",
            "dni",
            "relative_humidity",
        ],
}

geo_tmy_weather, geo_tmy_meta = pvdeg.weather.get(
    weather_db, geospatial=True, **weather_arg)

geo_tmy_meta = geo_tmy_meta[geo_tmy_meta["country"] == "United States"]
geo_tmy_meta = geo_tmy_meta[geo_tmy_meta["state"] != "Hawaii"]
geo_tmy_meta = geo_tmy_meta[geo_tmy_meta["state"] != "Alaska"]

sub_tmy_meta, sub_tmy_gids = pvdeg.utilities.gid_downsampling(geo_tmy_meta, 4)

geo_weather = geo_tmy_weather.assign_coords({'time': geo_tmy_weather.time - pd.Timedelta(minutes=30)})
geo_weather = geo_weather.sel(gid=sub_tmy_gids).chunk({"gid":10}) 

combined_gids = geo_weather.gid.values

target_dir = "/projects/pvsoiling/pvdeg/analysis/pvrw2025/diffuse_stow/"
step_size = 640 # chunks of 10 with 64 workers
for i in range(0, len(combined_gids), step_size):

    front = i
    back = min(i + step_size, len(combined_gids) - 1) # ensure we dont go out of bounds

    slice_weather = geo_weather.isel(gid=slice(front, back))
    slice_meta = sub_tmy_meta.iloc[front : back]

    partial_res = pvdeg.geospatial.analysis(
        weather_ds = slice_weather,
        meta_df = slice_meta,
        func = diffuse_stow
    )

    partial_res.to_netcdf(f"{target_dir}-{i}-{i+i-1}.nc")
    print("ended", i)