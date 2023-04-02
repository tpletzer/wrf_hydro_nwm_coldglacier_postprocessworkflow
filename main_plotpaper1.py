from matplotlib.collections import LineCollection
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.dates as mpd
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import defopt
import re
from scipy import interpolate
import sys
import math
sys.path.insert(1, '/nesi/project/uoo03104/.conda/envs/xesmf_stable_env/lib/python3.7/site-packages/cmcrameri/')
import cm
#add .plot(cmap=cm.hawaii) for cb friendly
import preprocess_xsect as prep
import datetime
#warnings.filterwarnings('ignore')
import plot_paper1_figs as ppf
import generate_albedo as alb

def dataframe_to_datetime(d):
    d['Datetime'] = pd.to_datetime(d['date'] + ' ' + d['hour'])
    d = d.set_index('Datetime')
    d = d.drop(['date','hour'], axis=1)
    d['date']=d.index
    return d


def plot_figs(*, save_dir: str='/nesi/project/uoo03104/Paper1_modelcomp', station_name: str='cwg', plot_name: str='runoffmod'):

    oldrunoff_oldalbedo = pd.read_csv(f'{save_dir}/oldrunoff_oldalb/NWM/timeseries_ldasout_cwg.csv',index_col=0)
    oldrunoff_oldalbedo.index = pd.to_datetime(oldrunoff_oldalbedo.index)
    oldrunoff_oldalbedo.index = oldrunoff_oldalbedo.index.tz_localize('UTC')

    newrunoff_newalbedo = pd.read_csv(f'{save_dir}/newrunoff_newalb/NWM/timeseries_ldasout_cwg.csv',index_col=0)
    newrunoff_newalbedo.index = pd.to_datetime(newrunoff_newalbedo.index)
    #newrunoff_newalbedo.index = newrunoff_newalbedo.index.tz_localize('UTC')

    newrunoff_oldalbedo = pd.read_csv(f'{save_dir}/newrunoff_oldalb/NWM/timeseries_ldasout_cwg.csv',index_col=0)
    newrunoff_oldalbedo.index = pd.to_datetime(newrunoff_oldalbedo.index)
    newrunoff_oldalbedo.index = newrunoff_oldalbedo.index.tz_localize('UTC')

    c = pd.read_csv('/nesi/project/uoo03104/validation_data/COHM_ASP_longerm_corrected.csv',delimiter=',',sep='\t')
    c = c.set_index('TIMESTAMP')
    c.index = pd.to_datetime(c.index)
    delta = datetime.timedelta(hours=13)
    c.index = (c.index - delta).tz_localize('UTC') #convert from NZDT to UTC
    c = c.astype(float)

    """
    Plan for figures (total 11 figs and 2 tables?):
    Pre-info:
    1. map of region DONE
    2. precip regression plot DONE
    3. cold glac vs og modifications for melt and runoff: DONE
    4. albedo comparison: obs vs old scheme vs new scheme (summer?? year???) DONE (ish)
    5. albedo comparison bands for old vs new scheme... DONE


    model runs:
    6. land surface temperature (scatter with RMSE) DONE
    7. temp xsection with thermacouples DONE
    8. hfx and some of the surface energy fluxes?? DONE
    snowheight over a year!??? DONE
    9. hourly runoff DONE



    10. AMPS vs obs forced model run input comparison temperature, windspeed, swdown, longwavedown, precip (scatter? and rmse)
    11. amps vs obs model: output comp ice temps, landsurface temp and runoff

    """
    if plot_name=="runoffmod":
        ppf.plot_runoffmod(oldrunoff_oldalbedo, newrunoff_oldalbedo, save_dir, station_name)

    if plot_name=="snowheight":
        ppf.plot_snowheight(oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo, c, save_dir, station_name)

    if plot_name=="runoffcomp":
        ppf.plot_runoffcomp(oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo, save_dir, station_name)

    if plot_name=="runoff3plot":
        ppf.plot_runoff3plot(oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo, save_dir, station_name)

    if plot_name=="sfctemptime":
        ppf.plot_sfctemptime(oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo, c, save_dir, station_name)

    if plot_name=="sfctempscatter":
        ppf.plot_sfctempscatter(oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo, c, save_dir, station_name)

    if plot_name=="albedobefore":
        ppf.plot_albedobefore(oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo, c, save_dir, station_name)

    if plot_name=="albedoafter":
        ppf.plot_albedoafter(oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo, c, save_dir, station_name)

    if plot_name=="albedobands":
        ppf.plot_albedobands(oldrunoff_oldalbedo,newrunoff_newalbedo, c, save_dir, station_name)

    if plot_name=="xsecticetemp":
        ppf.plot_xsecticetemp(newrunoff_newalbedo, c, save_dir, station_name)    

    if plot_name=="xsecticetempdiff":
        ppf.plot_xsecticetempdiff(newrunoff_newalbedo, c, save_dir, station_name)

    if plot_name=="xsecticetempspinnup":
        ppf.plot_xsecticetempspinnup(newrunoff_newalbedo, c, save_dir, station_name)

    if plot_name=="icetempH":
        ppf.plot_icetempH(newrunoff_newalbedo, c, save_dir, station_name)

if __name__ == "__main__":
    defopt.run(plot_figs)
