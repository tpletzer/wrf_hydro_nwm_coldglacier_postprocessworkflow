import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import re
from scipy import interpolate
import sys
sys.path.insert(1, '/nesi/project/uoo03104/.conda/envs/xesmf_stable_env/lib/python3.7/site-packages/cmcrameri/')
import cm

def proc_xsection(file_dir='.', station_name='cwg', var_name="PSNOWTEMP"):
    #df = pd.read_csv(f'{file_dir}/{station_name}_restart.csv', index_col=0)
    df = pd.read_csv(f'{file_dir}/timeseries_ldasout_{station_name}.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df_dz = df.filter(regex=r"PSNOWDZ")
    df_var = df.filter(regex=rf'{var_name}')
    df_snowh = df["SNOWH"] #glacier thickness at start of period
    #df_snowh.loc["2016-06-01 00:00:00"] = 50.0 #ini only
    df_dz[df_dz==0.]=np.nan
    df_var[df_var==0.]=np.nan

    d = df_dz

    d = -1*d
    d["PSNOWDZ0"] = df_snowh + d["PSNOWDZ0"]
    df_dz = d.cumsum(axis=1)
    df_dz = df_dz + df.filter(regex=r"PSNOWDZ")
    
    try:
        np.argwhere(np.isnan(df_dz.iloc[0].values))[0][0]
    except IndexError: #if its the init run
        drop_time = df_dz.iloc[0].name
        df_dz.drop([drop_time], axis=0, inplace=True)
        df_snowh.drop([drop_time], axis=0, inplace=True)
        df_var.drop([drop_time], axis=0, inplace=True)
   
    return df_snowh, df_dz, df_var




def preprocess_flow(save_dir='./', station_name='cwg'):
    df = pd.read_csv(f'{save_dir}/timeseries_ldasout_{station_name}.csv', index_col=0)
    df.index = pd.to_datetime(df.index)

    df = df[["PSNOWTHRUFAL", "FLOW_SNOW", "FLOW_ICE"]]

    return df


def proc_4panel(date="2016-12-01 00:00:00+00:00", save_dir='./', station_name='cwg'):
    df = pd.read_csv(f'{save_dir}/timeseries_ldasout_{station_name}.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df_dz = df.filter(regex=r"PSNOWDZ")
    df_temp = df.filter(regex=r"PSNOWTEMP")
    df_liq = df.filter(regex=r"PSNOWLIQ")
    df_rho = df.filter(regex=r"PSNOWRHO")
    df_heat = df.filter(regex=r"PSNOWHEAT")
    df_melt = df.filter(regex=r"PSNOWMELT")
    df_refrz = df.filter(regex=r"PSNOWREFRZ")

    df_dz[df_dz==0.]=np.nan
    df_temp[df_temp==0.]=np.nan
    df_heat[df_heat==0.]=np.nan
    df_rho[df_rho==999.]=np.nan
    df_liq[df_liq==0.]=np.nan
    df_melt[df_melt==0.]=np.nan
    df_refrz[df_refrz==0.]=np.nan

    #nan_index = np.argwhere(np.isnan(z_real))[0][0]
    #df_liq.where(np.isnan(df_dz)

    # d = df_dz
    # d = -1*d
    # df_snowh = df["SNOWH"]
    # d["PSNOWDZ0"] = df_snowh + d["PSNOWDZ0"]
    #d["PSNOWDZ0"] = -1*(d["PSNOWDZ0"].sub(df_snowh, axis=0))
    # df_dz = d.cumsum(axis=1)
    # df_dz = df_dz + df.filter(regex=r"PSNOWDZ")

    # height = df_dz.loc[date].values
    #height = np.linspace(0, 39, 40, dtype=int)
    
    height = np.linspace(0, len(df_temp.columns) - 1, len(df_temp.columns), dtype=int)
    temp = df_temp.loc[date].values
    heat = df_heat.loc[date].values
    rho = df_rho.loc[date].values
    liq = df_liq.loc[date].values
    melt = df_melt.loc[date].values
    refrz = df_refrz.loc[date].values

    thruf = df["PSNOWTHRUFAL"][date]
    fsno = df["FLOW_SNOW"][date]
    fice = df["FLOW_ICE"][date]

    print(f'{save_dir}: PSNOWTHRUFAL={thruf}, FLOW_SNOW={fsno}, FLOW_ICE={fice}')

    return height,temp,heat,rho,liq, thruf, fsno, fice, melt, refrz
