from matplotlib.collections import LineCollection
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
import generate_albedo as alb
from sklearn.neighbors import KNeighborsRegressor

#make colorblind friendly plots 
import seaborn as sns
from matplotlib.colors import ListedColormap
#cmap = ListedColormap(sns.color_palette('colorblind'))
#colors = iter([ListedColormap(sns.color_palette('colorblind'))(i) for i in range(10)])
colors = ListedColormap(sns.color_palette('colorblind'))

def dataframe_to_datetime(d):
    d['Datetime'] = pd.to_datetime(d['date'] + ' ' + d['hour'])
    d = d.set_index('Datetime')
    d = d.drop(['date','hour'], axis=1)
    d['date']=d.index
    return d

def plot_runoffmod(df_warm, df_cold, save_dir, station_name):
    dt = "2021-12-10 03:00:00+00:00"

    #cold_depths = [0., -0.02376214, -0.07584827, -0.22922483,  -0.3826014, -0.66388039]
    cold_depths = -1*df_cold.loc[dt].filter(regex=r'PSNOWDZ')[0:4].cumsum().values
    cold_depths = np.insert(cold_depths, [0], [0.], axis=0)
    cold = pd.DataFrame()
    liqs = df_cold.loc[dt].filter(regex=r'PSNOWLIQ')[0:5].values
    liqs = np.insert(liqs, [0], [liqs[0]], axis=0)
    cold["liq"] = liqs
    melts = df_cold.loc[dt].filter(regex=r'PSNOWMELT')[0:5].values
    melts = np.insert(melts, [0], [melts[0]], axis=0)
    cold["melt"] = melts
    refrzs = df_cold.loc[dt].filter(regex=r'PSNOWREFRZ')[0:5].values
    refrzs = np.insert(refrzs, [0], [refrzs[0]], axis=0)
    cold["refrz"] = refrzs

    #warm_depths = [0., -0.02322585, -0.07531198, -0.22868854, -0.38206511, -0.6633441 ]
    warm_depths = -1*df_warm.loc[dt].filter(regex=r'PSNOWDZ')[0:4].cumsum().values
    warm_depths = np.insert(warm_depths, [0], [0.], axis=0)
    warm = pd.DataFrame()
    liqs = df_warm.loc[dt].filter(regex=r'PSNOWLIQ')[0:5].values
    liqs = np.insert(liqs, [0], [liqs[0]], axis=0) #do the first layer twice for stairs
    warm["liq"] = liqs
    melts = df_warm.loc[dt].filter(regex=r'PSNOWMELT')[0:5].values
    melts = np.insert(melts, [0], [melts[0]], axis=0)
    warm["melt"] = melts
    refrzs = df_warm.loc[dt].filter(regex=r'PSNOWREFRZ')[0:5].values
    refrzs = np.insert(refrzs, [0], [refrzs[0]], axis=0)
    warm["refrz"] = refrzs

    ftsize=14
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10,6))
    fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical')
    fig.text(0.45, 0.04, 'Amount of water per layer (mm)', va='center', rotation='horizontal')

    #warmglac
    fig.text(0.2, 0.90, f'oldrunoff: {df_warm.loc[dt]["PSNOWTHRUFAL"]} mm of runoff', va='center', rotation='horizontal')
    axs[0].stairs(warm_depths, warm["liq"]*1000, orientation='vertical', baseline=None,linestyle='-', color='b',label='liquid water content', linewidth=1.5)
    axs[0].stairs(warm_depths, warm["melt"]*1000, orientation='vertical', baseline=None,linestyle='-', color='r',label='meltwater', linewidth=1.5)
    axs[0].stairs(warm_depths, warm["refrz"]*1000, orientation='vertical', baseline=None, linestyle='-', color='cyan',label='refrozen', linewidth=1.5)
    axs[0].legend(loc='lower right')
    axs[0].set_ylim(-0.4, 0.)
    axs[0].axhline(-0.03272384, color='k', linestyle='dotted', linewidth=0.7)
    axs[0].axhline(-0.08480997, color='k', linestyle='dotted', linewidth=0.7)
    axs[0].axhline(-0.23818653, color='k', linestyle='dotted', linewidth=0.7)
    axs[0].axhspan(-0.4,-0.03272384, alpha=0.2)

    #colglac
    fig.text(0.62, 0.90, f'newrunoff: {np.round(df_cold.loc[dt]["PSNOWTHRUFAL"], 2)} mm of runoff', va='center', rotation='horizontal')
    axs[1].stairs(cold_depths, cold["liq"]*1000, orientation='vertical', baseline=None,linestyle='-', color='b',label='liquid water content', linewidth=1.5)
    axs[1].stairs(cold_depths, cold["melt"]*1000, orientation='vertical', baseline=None,linestyle='-', color='r',label='meltwater', linewidth=1.5)
    axs[1].stairs(cold_depths, cold["refrz"]*1000, orientation='vertical', baseline=None, linestyle='-', color='cyan',label='refrozen', linewidth=1.5)
    axs[1].legend(loc='lower right')
    axs[1].axhline(-0.03272384, color='k', linestyle='dotted', linewidth=0.7)
    axs[1].axhline(-0.08480997, color='k', linestyle='dotted', linewidth=0.7)
    axs[1].axhline(-0.23818653, color='k', linestyle='dotted', linewidth=0.7)
    axs[1].axhspan(-0.4,-0.03272384, alpha=0.2)

    plt.savefig(f'{save_dir}/runoffmod_{station_name}.png', bbox_inches='tight')

def plot_snowheight(oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo, c, save_dir, station_name):
    hsnow_oo = pd.DataFrame()
    hsnow_oo["SNOWH"] = oldrunoff_oldalbedo["SNOWH"].loc['2021-12-01 03:00:00+00:00':]
    hsnow_oo["SNOWH_scaled"] = hsnow_oo["SNOWH"] - hsnow_oo["SNOWH"][0]

    hsnow_no = pd.DataFrame()
    hsnow_no["SNOWH"] = newrunoff_oldalbedo["SNOWH"].loc['2021-12-01 03:00:00+00:00':oldrunoff_oldalbedo.index[-1]]
    hsnow_no["SNOWH_scaled"] = hsnow_no["SNOWH"] - hsnow_no["SNOWH"][0]

    hsnow_nn = pd.DataFrame()
    hsnow_nn["SNOWH"] = newrunoff_newalbedo["SNOWH"].loc['2021-12-01 03:00:00+00:00':oldrunoff_oldalbedo.index[-1]]
    hsnow_nn["SNOWH_scaled"] = hsnow_nn["SNOWH"] - hsnow_nn["SNOWH"][0]

    hsnow = pd.DataFrame()
    hsnow["SR50T"] = -1*c["SR50T_Avg"].loc[c.index >= '2021-12-01 03:00:00+00:00']
    hsnow["hsnow"] = hsnow["SR50T"] - hsnow["SR50T"][0]
    clf = KNeighborsRegressor(n_neighbors=30, weights='uniform') #do double since its 15min instead of 30
    clf.fit(hsnow.index.values[:, np.newaxis], hsnow.hsnow.values)
    y_pred = clf.predict(hsnow.index.values[:, np.newaxis])
    hsnow["ypred"] = y_pred

    plt.figure(figsize=[15, 7])
    hsnow_oo["SNOWH_scaled"].plot(label='oldrunoff_oldalbedo', color=colors(0))
    hsnow_no["SNOWH_scaled"].plot(label='newrunoff_oldalbedo', color=colors(2))
    hsnow_nn["SNOWH_scaled"].plot(label='newrunoff_newalbedo', color=colors(3))
    hsnow.ypred[:hsnow_oo.index[-1]].plot(label='observed', color=colors(7))
    plt.xlabel('')
    plt.ylabel('surface height (m)')
    plt.legend(loc='upper right')
    plt.savefig(f'{save_dir}/snowheight_{station_name}.png', bbox_inches='tight')


def plot_runoffcomp(oldrunoff_oldalbedo,newrunoff_oldalbedo,newrunoff_newalbedo, save_dir, station_name):
    plt.figure(figsize=[12, 7])
    oldrunoff_oldalbedo["PSNOWTHRUFAL"].loc['2021-12-01 01:00:00+00:00':oldrunoff_oldalbedo.index[-1]].plot(label='oldrunoff_oldalbedo', color=colors(0))
    newrunoff_oldalbedo["PSNOWTHRUFAL"].loc['2021-12-01 01:00:00+00:00':oldrunoff_oldalbedo.index[-1]].plot(label='newrunoff_oldalbedo', color=colors(2))
    newrunoff_newalbedo["PSNOWTHRUFAL"].loc['2021-12-01 01:00:00+00:00':oldrunoff_oldalbedo.index[-1]].plot(label='newrunoff_newalbedo', color=colors(3))
    plt.ylabel('Runoff (mm)')
    plt.legend(loc='upper right')
    plt.savefig(f'{save_dir}/runoffcomp_{station_name}.png', bbox_inches='tight')

def plot_runoff3plot(oldrunoff_oldalbedo,newrunoff_oldalbedo,newrunoff_newalbedo, save_dir, station_name):
    run_list = [oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo]
    run_name = ["oldrunoff_oldalbedo", "newrunoff_oldalbedo", "newrunoff_newalbedo"]
    color_list = [colors(0), colors(2), colors(3)]
    fig, axs = plt.subplots(1,3,figsize=(15, 5), sharex=True, sharey=True)
    fig.text(0.08, 0.5, 'Runoff (mm)', va='center', rotation='vertical', fontsize=14)
    for i in range(len(run_list)):
        run_list[i]["PSNOWTHRUFAL"].loc['2021-12-01 01:00:00+00:00':oldrunoff_oldalbedo.index[-1]].plot(label=run_name[i], color=color_list[i], ax=axs[i])
        axs[i].legend(loc='upper center', fontsize=13)
    plt.savefig(f'{save_dir}/runoff3plot_{station_name}.png', bbox_inches='tight')

def plot_sfctemptime(oldrunoff_oldalbedo,newrunoff_oldalbedo,newrunoff_newalbedo, c, save_dir, station_name):
    
    lwd = c["incomingLW_Avg"].loc["2021-12-14 00:00:00+00:00":"2021-12-31 23:00:00+00:00"]
    lwu = c["outgoingLW_Avg"].loc["2021-12-14 00:00:00+00:00":"2021-12-31 23:00:00+00:00"]
    airT = c["AirTC_Avg"] + 273.15
    airT = airT.loc["2021-12-14 00:00:00+00:00":"2021-12-31 23:00:00+00:00"]
    sigma = 5.67e-8
    eps = 0.98
    tsfc_1 = ((1/sigma)*lwu)**(0.25)
    tsfc_98 = ((1/(eps*sigma))*(lwu - lwd + (eps*lwd)))**(0.25)
    icetemp = tsfc_1.resample('H').mean()
    icetemp[icetemp>273.15] = 273.15 #to match obs to croc

    plt.figure(figsize=[15,7])
    oldrunoff_oldalbedo["TG"].loc["2021-12-14 00:00:00+00:00":"2021-12-31 23:00:00+00:00"].plot(label='oldrunoff_oldalbedo', color=colors(0), linewidth=1)
    newrunoff_oldalbedo["TG"].loc["2021-12-14 00:00:00+00:00":"2021-12-31 23:00:00+00:00"].plot(label='newrunoff_oldalbedo', color=colors(2), linewidth=1)
    newrunoff_newalbedo["TG"].loc["2021-12-14 00:00:00+00:00":"2021-12-31 23:00:00+00:00"].plot(label='newrunoff_newalbedo', color=colors(3), linewidth=1)
    icetemp.plot(label="observed", linewidth=1, color=colors(7))
    plt.ylabel('Surface temperature (K)')
    plt.xlabel('')
    plt.legend(loc="lower left")
    plt.savefig(f'{save_dir}/sfctemptime_{station_name}.png')

def plot_sfctempscatter(oldrunoff_oldalbedo,newrunoff_oldalbedo,newrunoff_newalbedo, c, save_dir, station_name):
    lwd = c["incomingLW_Avg"].loc["2021-12-01 03:00:00+00:00":"2022-02-28 23:00:00+00:00"]
    lwu = c["outgoingLW_Avg"].loc["2021-12-01 03:00:00+00:00":"2022-02-28 23:00:00+00:00"]
    airT = c["AirTC_Avg"] + 273.15
    airT = airT.loc["2021-12-01 03:00:00+00:00":"2022-02-28 23:00:00+00:00"]
    sigma = 5.67e-8
    eps = 0.98
    tsfc_1 = ((1/sigma)*lwu)**(0.25)
    tsfc_98 = ((1/(eps*sigma))*(lwu - lwd + (eps*lwd)))**(0.25)
    icetemp = tsfc_1.resample('H').mean()
    icetemp[icetemp>273.15] = 273.15 #to match obs to croc

    RMSE_list = []
    run_list = [oldrunoff_oldalbedo, newrunoff_oldalbedo, newrunoff_newalbedo]
    run_name = ["oldrunoff_oldalbedo", "newrunoff_oldalbedo", "newrunoff_newalbedo"]
    color_list = [colors(0), colors(2), colors(3)]
    fig, axs = plt.subplots(1,3,figsize=(15, 5))
    fig.text(0.08, 0.5, 'Modelled surface temperature (K)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.4, 0.07, 'Observed surface temperature (K)', va='center', rotation='horizontal', fontsize=14)
    #RMSE and MSE:
    for i in range(len(run_list)):
        MSE = np.square(np.subtract(icetemp,run_list[i]["TG"].loc[icetemp.index[0]:])).mean()
        RMSE_list.append(math.sqrt(MSE))
        axs[i].scatter(icetemp, run_list[i]["TG"].loc[icetemp.index[0]:], s=20, facecolors='none', edgecolors=color_list[i],label=run_name[i])
        axs[i].plot([], [], ' ', label=f'RMSE = {np.round(RMSE_list[i],2)}')
        lims = [
            np.min([axs[0].get_xlim(), axs[0].get_ylim()]),  # min of both axes
        np.max([axs[0].get_xlim(), axs[0].get_ylim()]),  # max of both axes
        ]
        axs[i].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        axs[i].set_aspect('equal')
        axs[i].set_xlim(lims)
        axs[i].set_ylim(lims)
        axs[i].legend(loc='lower center', fontsize=13)
    plt.savefig(f'{save_dir}/sfctempscatter_{station_name}.png')



def plot_albedoafter(oldrunoff_oldalbedo,newrunoff_oldalbedo,newrunoff_newalbedo, c, save_dir, station_name):
    SWout = c["outgoingSW_Avg"].rolling(48,center=True).sum().ffill()
    SWin = c["incommingSW_Avg"].rolling(48,center=True).sum().ffill()
    Alb = SWout/SWin
    Alb = Alb["2021-12-02 00:00:00+00:00":]

    SWin_2hravg = (c["incommingSW_Avg"].resample('2H').mean()).resample('D').sum() #Van Den Broeke acc alb 2hr mean
    SWout_2hravg = (c["outgoingSW_Avg"].resample('2H').mean()).resample('D').sum()
    Alb_2hravg = SWout_2hravg /SWin_2hravg 
    Alb_2hravg = Alb_2hravg["2021-12-02 00:00:00+00:00":]

    fig = plt.figure(figsize=(15,5))
    oldrunoff_oldalbedo.ALBEDO[c.index.min():].plot(color=colors(0), label='oldalbedo')
    newrunoff_newalbedo.ALBEDO[c.index.min():].plot(color=colors(3), label='newalbedo')
    #c.alb[:"2022-02-28 23:00:00+00:00"].plot(color=colors(7), label='observed') #acc alb
    Alb[:"2022-02-28 23:00:00+00:00"].plot(color=colors(7), label='observed') # 2hr mean daily acc alb (Van den Broeke 2004)
    plt.legend(loc='upper right')
    plt.xlabel('')
    plt.ylabel('Albedo')
    #plt.show()
    plt.savefig(f'{save_dir}/albedoafter_{station_name}.png', bbox_inches='tight')

def plot_albedobefore(oldrunoff_oldalbedo,newrunoff_oldalbedo,newrunoff_newalbedo, c, save_dir, station_name):
    SWout = c["outgoingSW_Avg"].rolling(48,center=True).sum().ffill()
    SWin = c["incommingSW_Avg"].rolling(48,center=True).sum().ffill()
    Alb = SWout/SWin
    Alb = Alb["2021-12-02 00:00:00+00:00":]

    SWin_2hravg = (c["incommingSW_Avg"].resample('2H').mean()).resample('D').sum() #Van Den Broeke acc alb 2hr mean
    SWout_2hravg = (c["outgoingSW_Avg"].resample('2H').mean()).resample('D').sum()
    Alb_2hravg = SWout_2hravg /SWin_2hravg
    Alb_2hravg = Alb_2hravg["2021-12-02 00:00:00+00:00":]

    fig = plt.figure(figsize=(15,5))
    oldrunoff_oldalbedo.ALBEDO[c.index.min():].plot(color=colors(0), label='oldalbedo')
    #c.alb[:"2022-02-28 23:00:00+00:00"].plot(color=colors(7), label='observed')
    Alb[:"2022-02-28 23:00:00+00:00"].plot(color=colors(7), label='observed') # 2hr mean daily acc alb (Van den Broeke 2004)
    plt.legend(loc='upper right')
    plt.xlabel('')
    plt.ylabel('Albedo')
    #plt.show()
    plt.savefig(f'{save_dir}/albedobefore_{station_name}.png', bbox_inches='tight')

def plot_albedobands(oldrunoff_oldalbedo,newrunoff_newalbedo, c, save_dir, station_name):

    newalb_df = newrunoff_newalbedo[["PSNOWALB", "PSNOWGRAN1_0", "PSNOWGRAN1_1", "PSNOWGRAN2_0", "PSNOWGRAN2_1", "PSNOWDZ0", "PSNOWDZ1", "PSNOWAGE0", "PSNOWAGE1", "PSNOWRHO0","PSNOWRHO1"]]
    newalb_df["P"] = c["Air_pressure_Avg"]*100.
    newalb_df["PSNOWRHO01"] = newalb_df[["PSNOWRHO0","PSNOWRHO1"]].values.tolist()
    newalb_df["P"]["2021-12-01 00:00:00+00:00":"2021-12-01 02:00:00+00:00"] = 95000.0
    new_xalb_dict = {0: [0.96, 1.58, .92, .90, 15.4, 346.3, 32.31, .88, .200, .65, 0.3],
                1: [0.96, 1.58, .94, .66, 15.4, 346.3, 21., .35, .200, .89, 0.31], #this one
                2: [0.74, 1.58, .72, .97, 15.4, 346.3, 32.31, .88, .200, .74, 0.7]
                }
    new_xvalb_df, new_xvalb_band_df, new_zdiam_df, new_zalb_top_df, new_zalb_bot_df = alb.snowcroalb_3param(new_xalb_dict, newalb_df, 'newalbedo')

    oldalb_df = oldrunoff_oldalbedo
    oldalb_df = oldalb_df[["PSNOWALB", "PSNOWGRAN1_0", "PSNOWGRAN1_1", "PSNOWGRAN2_0", "PSNOWGRAN2_1", "PSNOWDZ0", "PSNOWDZ1", "PSNOWAGE0", "PSNOWAGE1", "PSNOWRHO0","PSNOWRHO1"]]
    oldalb_df["P"] = c["Air_pressure_Avg"]*100.
    oldalb_df["PSNOWRHO01"] = oldalb_df[["PSNOWRHO0","PSNOWRHO1"]].values.tolist()
    oldalb_df["P"]["2021-12-01 00:00:00+00:00":"2021-12-01 02:00:00+00:00"] = 95000.0
    old_xalb_dict = {0: [0.96, 1.58, .92, .90, 15.4, 346.3, 32.31, .88, .200, .65, 0.3], #this one
                1: [0.96, 1.58, .94, .66, 15.4, 346.3, 21., .35, .200, .89, 0.31],
                2: [0.74, 1.58, .72, .97, 15.4, 346.3, 32.31, .88, .200, .74, 0.7]
                }
    old_xvalb_df, old_xvalb_band_df, old_zdiam_df, old_zalb_top_df, old_zalb_bot_df = alb.snowcroalb_3param(old_xalb_dict, oldalb_df, 'oldalbedo')

    SWout = c["outgoingSW_Avg"].rolling(48,center=True).sum().ffill()
    SWin = c["incommingSW_Avg"].rolling(48,center=True).sum().ffill()
    Alb = SWout/SWin
    Alb = Alb["2021-12-02 00:00:00+00:00":]

    SWin_2hravg = (c["incommingSW_Avg"].resample('2H').mean()).resample('D').sum() #Van Den Broeke acc alb 2hr mean
    SWout_2hravg = (c["outgoingSW_Avg"].resample('2H').mean()).resample('D').sum()
    Alb_2hravg = SWout_2hravg /SWin_2hravg 
    Alb_2hravg = Alb_2hravg["2021-12-02 00:00:00+00:00":]

    ftsize=12
    fig, axs = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(12,8))
    newalb_df["PSNOWALB"].loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[0], color=colors(3), label=f'Broadband newalbedo')
    new_xvalb_band_df["1_1"].loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[0], color=colors(3), linestyle='--', label=f'Band 1 newalbedo')
    new_xvalb_band_df["1_2"].loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[0], color=colors(3), linestyle=':', label=f'Band 2 newalbedo')
    new_xvalb_band_df["1_3"].loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[0], color=colors(3), linestyle='-.', label=f'Band 3 newalbedo')
    #c.alb.loc[:"2022-02-28 23:00:00+00:00"].plot(ax=axs[0], color=colors(7), label=f'Broadband oberserved')
    Alb.loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[0], color=colors(7), label=f'Broadband oberserved')
    axs[0].legend(loc='lower center')
    axs[0].set_ylabel('Albedo')
    axs[0].set_ylim(0.0,1.0)
    oldalb_df["PSNOWALB"].loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[1], color=colors(0), label=f'Broadband oldalbedo')
    old_xvalb_band_df["0_1"].loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[1], color=colors(0), linestyle='--', label=f'Band 1 oldalbedo')
    old_xvalb_band_df["0_2"].loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[1], color=colors(0), linestyle=':', label=f'Band 2 oldalbedo')
    old_xvalb_band_df["0_3"].loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[1], color=colors(0), linestyle='-.', label=f'Band 3 oldalbedo')
    #c.alb.loc[:"2022-02-28 23:00:00+00:00"].plot(ax=axs[1], color=colors(7), label=f'Broadband oberserved')
    Alb.loc["2021-12-01 00:00:00+00:00":"2022-02-28 23:00:00+00:00"].plot(ax=axs[1], color=colors(7), label=f'Broadband oberserved')
    axs[1].legend(loc='lower center')
    axs[1].set_ylabel('Albedo')
    plt.xlabel('')
    plt.savefig(f'{save_dir}/albedobands_{station_name}.png')

def plot_xsecticetemp(newrunoff_newalbedo, c, save_dir, station_name):

    """
    function to plot cross section of modelled glacier temps with thermacouples inside
    TO Do:
    cut to before influx of melt

    """
    hsnow_m = pd.DataFrame()
    hsnow_m["SNOWH"] = newrunoff_newalbedo["SNOWH"].loc["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
    #hsnow_m["SNOWH"] = df["SNOWH"].loc["2021-12-01 13:00:00":"2021-12-15 00:00:00"]
    hsnow_m["SNOWH_scaled"] = hsnow_m["SNOWH"] - hsnow_m["SNOWH"][0]

    hsnow = pd.DataFrame()
    hsnow["SR50T"] = -1*c["SR50T_Avg"].loc["2021-12-01 03:00:00+00":"2021-12-15 00:00:00+00"]
    hsnow["hsnow"] = hsnow["SR50T"] - hsnow["SR50T"][0]

    hsnow = hsnow.resample('H').mean()
    hsnow["h_TC1"] = 0.05 + hsnow["hsnow"] #calculate height of sensor as snowpack melts
    hsnow["h_TC2"] = 0.1 + hsnow["hsnow"]
    hsnow["h_TC3"] = 0.2 + hsnow["hsnow"]
    hsnow["h_TC4"] = 0.5 + hsnow["hsnow"]
    hsnow["h_TC5"] = 1.0 + hsnow["hsnow"]
    hsnow["h_TC6"] = 2.0 + hsnow["hsnow"]

    c["depth"] = 2.0
    c = c.resample('H').mean()
    c = c.loc["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
    df_test = c[["TC1_Avg", "depth"]]#.loc["2021-12-01 13:00:00":"2021-12-15 00:00:00"] #"2022-01-01 01:00:00"
    # df_test = df_test.loc["2021-12-01 13:00:00+00:00":]

    var_names = ["PSNOWTEMP", "PSNOWRHO", "PSNOWLIQ", "PSNOWHEAT"]
    Z_list = []
    cp_list = []

    for var_name in var_names[:1]:
        df_snowh, df_dz, df_var = prep.proc_xsection(save_dir+'/newrunoff_newalb/NWM', station_name, var_name)
        df_snowh = df_snowh["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
        df_dz = df_dz["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
        df_var = df_var["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
        z = np.arange(df_snowh.values.max() - 0.1, df_snowh.values.max(), 0.001)
        #z = np.arange(df_snowh.values.max() - 0.5, df_snowh.values.max(), 0.01)
        z = np.append(np.arange(df_snowh.values.max() - 1.0, df_snowh.values.max() - 0.1, 0.5), z)
        z = np.append(np.arange(df_snowh.values.max() - 2.1, df_snowh.values.max() - 1.0, 0.5), z)
        #z = np.append(np.arange(df_snowh.values.max() - 3.5, df_snowh.values.max() - 0.5, 0.5), z)
        z.sort()
        z_rev = z

        dt = pd.DataFrame(columns=["depths", "var"], index=df_dz.index)
        dt2 = pd.DataFrame(columns=z_rev, index=df_dz.index)

        for index, row in df_dz.iterrows(): #iterating over all of the timesteps
            z_real = df_dz.loc[index].to_list() #extract the depth
            #find index of the first element isnan
            try:
                nan_index = np.argwhere(np.isnan(z_real))[0][0]
            except IndexError:
                nan_index = -1 #this only happens when init run
            #interp wants monotonically increasing z_real and no NaNs
            z_real_rev = z_real[0:nan_index]
            z_real_rev.reverse()
            t_real_rev = df_var.loc[index][0:nan_index].to_list()
            t_real_rev.reverse() #reverse is an in place operation

            f = interpolate.interp1d(z_real_rev, t_real_rev, bounds_error=False)
            t = f(z_rev)
            dt.loc[index] = pd.Series({'depths':z_rev, 'var':t})
            for ind in range(len(t)):
                dt2.loc[index][z_rev[ind]] = t[ind]

        data=dt2["2021-12-01 03:00:00":"2021-12-15 00:00:00"]
        x_vals = np.linspace(0, len(data.index), len(data.index), dtype=int)
        y_vals = z_rev
        # y_vals = np.linspace(0, len(z), len(z), dtype=int)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        Z = data.values
        Z_list.append(Z)

    ftsize=14
    fig = plt.figure(figsize=(15,10))
    axs = fig.add_subplot(1, 1, 1)
    var_dict = {
    'PSNOWTEMP': [1, cm.lajolla, 100, "Temperature", "K", [273.15]],  #[levels, cmap, nbins, label, unit]
    'PSNOWRHO': [1, cm.hawaii, 100, "Density", "kg/m3", [850]],
    'PSNOWLIQ': [1, cm.vik, 100, "Liquid content", "mmwe", [0]],
    'PSNOWHEAT': [1, cm.vik, 100, "Heat content", "J/m2", [0]],
    }
    x = Z_list[0][~pd.isnull(Z_list[0])]
    min_z = x.min()
    max_z = x.max()
    step_z = (max_z - min_z)/100.
    lev = np.arange(min_z, max_z + (2*step_z), step_z)
    cp = axs.contourf(X, Y, Z_list[0], cmap=var_dict[var_names[0]][1], norm=plt.Normalize(252., 274.), levels=lev)   #, levels=var_dict[var_names[r]][0])
    fig.colorbar(cp, ax=axs, label=f'{var_dict[var_names[0]][3]} ({var_dict[var_names[0]][4]})')
    axs.plot(X, df_snowh[:"2021-12-15 00:00:00"].values, '-k', linewidth=0.1)
    #axs.set_ylim([48.0, Y.max()])

    for i in hsnow.filter(regex=r'h_').columns:
        i_c = i.split('_')[1]
        df_test = c[[f'{i_c}_Avg', 'depth']]#.loc["2021-12-01 00:00:00":"2021-12-15 00:00:00"]
        # df_test = df_test.loc["2021-12-01 13:00:00+00:00":]
        df_test["depth"] = (hsnow_m["SNOWH"][0] - hsnow[i])#.loc["2021-12-01 13:00:00+00:00":"2022-01-01 01:00:00+00:00"]
        df_test[f'{i_c}_Avg'] = df_test[f'{i_c}_Avg']+273.15
        x_c = x_vals
        y_c=df_test["depth"].values
        col=df_test[f'{i_c}_Avg'].values
        #plots every 5th dot
        axs.scatter(x_c[::5], y_c[::5], c=col[::5], cmap=var_dict[var_names[0]][1], norm=plt.Normalize(252., 274.), marker='o', edgecolors='white', linewidth=0.2) #, marker='o', edgecolors='grey') #, levels=lev)

    my_xticks = []
    for i in data.index.values:
        my_xticks.append(i)
    my_xticks2 = [re.sub(r'\w\d\d\:00\:00\.0+$', '', str(d)) for d in my_xticks]

    plt.xticks(list(range(0,len(data.index),1)), my_xticks2, rotation=45)
    n=var_dict["PSNOWTEMP"][2]
    plt.locator_params(axis='x', nbins=len(X)/24.)
    plt.subplots_adjust(top=0.934, bottom=0.145, left=0.063, right=0.99, hspace=0.19, wspace=0.2)
    fig.supylabel('Height (m)', fontsize=12)
    plt.ylim(48.9, 50.1)
    # plt.show()
    plt.savefig(f'{save_dir}/xsecticetemp_{station_name}.png', bbox_inches='tight')

def plot_icetempH(newrunoff_newalbedo, c, save_dir, station_name):
    df_snowh, df_dz, df_var = prep.proc_xsection(save_dir+'/newrunoff_newalb/NWM/', station_name, "PSNOWTEMP")

    hsnow = pd.DataFrame()
    hsnow["SR50T"] = -1*c["SR50T_Avg"].loc["2021-12-01 03:00:00+00":"2021-12-15 00:00:00+00"]
    hsnow["hsnow"] = hsnow["SR50T"] - hsnow["SR50T"][0]

    hsnow = hsnow.resample('H').mean()
    hsnow["h_TC1"] = 0.05 + hsnow["hsnow"] #calculate height of sensor as snowpack melts
    hsnow["h_TC2"] = 0.1 + hsnow["hsnow"]
    hsnow["h_TC3"] = 0.2 + hsnow["hsnow"]
    hsnow["h_TC4"] = 0.5 + hsnow["hsnow"]
    hsnow["h_TC5"] = 1.0 + hsnow["hsnow"]
    hsnow["h_TC6"] = 2.0 + hsnow["hsnow"]

    df_dz = df_dz.loc[hsnow.index[0]:]
    df_var = df_var.loc[hsnow.index[0]:]
    df_snowh = df_snowh.loc[hsnow.index[0]:]

    z_005 = df_snowh - hsnow["h_TC1"] #snow height of each sensor mapped to model
    z_010 = df_snowh - hsnow["h_TC2"]
    z_020 = df_snowh - hsnow["h_TC3"]
    z_050 = df_snowh - hsnow["h_TC4"]
    z_100 = df_snowh - hsnow["h_TC5"]
    z_200 = df_snowh - hsnow["h_TC6"]

    #interpolate for each sensor height in model
    dt = pd.DataFrame(columns=["0.05", "0.1", "0.2", "0.5", "1.0", "2.0"], index=z_005.index)
    z_005 = z_005.loc[:df_snowh.index[-1]]
    for i in range(len(z_005)):
        # print(z_005.iloc[i]) #target to interp to
        f = interpolate.interp1d(df_dz.iloc[i].values, df_var.iloc[i].values, bounds_error=False)
        t_005 = f(z_005.iloc[i])
        t_010 = f(z_010.iloc[i])
        t_020 = f(z_020.iloc[i])
        t_050 = f(z_050.iloc[i])
        t_100 = f(z_100.iloc[i])
        t_200 = f(z_200.iloc[i])
        dt.iloc[i] = pd.Series({'0.05':t_005, '0.1':t_010, '0.2':t_020, '0.5':t_050, '1.0':t_100, '2.0':t_200}, dtype=np.float64)
    dt = dt.astype(np.float64)
    dt = dt-273.15

    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']

    RMSE_list = []
    run_list = [dt["0.05"], dt["0.1"], dt["0.2"], dt["0.5"], dt["1.0"], dt["2.0"]]
    run_name = ["TC1", "TC2", "TC3", "TC4", "TC5", "TC6"]
    run_listobs = [c["TC1_Avg"], c["TC2_Avg"], c["TC3_Avg"], c["TC4_Avg"], c["TC5_Avg"], c["TC6_Avg"]]

    fig, ax1 = plt.subplots(figsize=(12,8))
    c["TC1_Avg"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='red', linestyle='dotted', legend=False, label='_TC1')
    c["TC2_Avg"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='orange', linestyle='dotted', legend=False, label='_TC2')
    c["TC3_Avg"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='green', linestyle='dotted', legend=False, label='_TC3')
    c["TC4_Avg"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='blue', linestyle='dotted', legend=False, label='_TC4')
    c["TC5_Avg"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='purple', linestyle='dotted', legend=False, label='_TC5')
    c["TC6_Avg"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='brown', linestyle='dotted', legend=False, label='_TC6')
    dt["0.05"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='red', linestyle='dashed', legend=False, label='_TC1mod')
    dt["0.1"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='orange', linestyle='dashed', legend=False, label='_TC2mod')
    dt["0.2"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='green',linestyle='dashed', legend=False, label='_TC3mod')
    dt["0.5"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='blue', linestyle='dashed', legend=False, label='_TC4mod')
    dt["1.0"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='purple', linestyle='dashed', legend=False, label='_TC5mod')
    dt["2.0"].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"].plot(ax=ax1, color='brown', linestyle='dashed', legend=False, label='_TC6mod')
    ax1.plot([], [],' ', label='RMSE:')
    for i in range(len(run_list)):
        MSE = np.square(np.subtract(run_listobs[i].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"],run_list[i].loc["2021-12-01 03:00:00":"2021-12-15 00:00:00"])).mean()
        RMSE_list.append(math.sqrt(MSE))
        ax1.plot([], [],' ',label=f'{run_name[i]} = {np.round(RMSE_list[i],2)}')
    plt.legend(loc='upper left')
    plt.ylabel('Temperature(C)')
    plt.xlabel('')
    pos = ax1.twinx()
    for i in range(len(colors)):
        pos.plot([], color=colors[i], label=f'TC{i+1}')
    pos.plot([], linestyle='dashed', color='k', label='newrunoff_newalbedo')
    pos.plot([], linestyle='dotted', color='k', label='observed')
    pos.tick_params(axis='y', left=False, right=False,labelleft=False, labelright=False)
    pos.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -.16))
    plt.tight_layout()
    plt.savefig(f'{save_dir}/icetempH_{station_name}.png')

def plot_xsecticetempdiff(newrunoff_newalbedo, c, save_dir, station_name):

    """
    function to plot cross section of modelled glacier temps with thermacouples inside

    """
    hsnow_m = pd.DataFrame()
    hsnow_m["SNOWH"] = newrunoff_newalbedo["SNOWH"].loc["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
    #hsnow_m["SNOWH"] = df["SNOWH"].loc["2021-12-01 13:00:00":"2021-12-15 00:00:00"]
    hsnow_m["SNOWH_scaled"] = hsnow_m["SNOWH"] - hsnow_m["SNOWH"][0]

    hsnow = pd.DataFrame()
    hsnow["SR50T"] = -1*c["SR50T_Avg"].loc["2021-12-01 03:00:00+00":"2021-12-15 00:00:00+00"]
    hsnow["hsnow"] = hsnow["SR50T"] - hsnow["SR50T"][0]

    hsnow = hsnow.resample('H').mean()
    hsnow["h_TC1"] = 0.05 + hsnow["hsnow"] #calculate height of sensor as snowpack melts
    hsnow["h_TC2"] = 0.1 + hsnow["hsnow"]
    hsnow["h_TC3"] = 0.2 + hsnow["hsnow"]
    hsnow["h_TC4"] = 0.5 + hsnow["hsnow"]
    hsnow["h_TC5"] = 1.0 + hsnow["hsnow"]
    hsnow["h_TC6"] = 2.0 + hsnow["hsnow"]

    c["depth"] = 2.0
    c = c.resample('H').mean()
    c = c.loc["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
    df_test = c[["TC1_Avg", "depth"]]#.loc["2021-12-01 13:00:00":"2021-12-15 00:00:00"] #"2022-01-01 01:00:00"
    # df_test = df_test.loc["2021-12-01 13:00:00+00:00":]

    var_names = ["PSNOWTEMP", "PSNOWRHO", "PSNOWLIQ", "PSNOWHEAT"]
    Z_list = []
    cp_list = []

    for var_name in var_names[:1]:
        df_snowh, df_dz, df_var = prep.proc_xsection(save_dir+'/newrunoff_newalb/NWM', station_name, var_name)
        df_snowh = df_snowh["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
        df_dz = df_dz["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
        df_var = df_var["2021-12-01 03:00:00+00:00":"2021-12-15 00:00:00+00:00"]
        z = np.arange(df_snowh.values.max() - 0.1, df_snowh.values.max(), 0.001)
        #z = np.arange(df_snowh.values.max() - 0.5, df_snowh.values.max(), 0.01)
        z = np.append(np.arange(df_snowh.values.max() - 1.0, df_snowh.values.max() - 0.1, 0.5), z)
        z = np.append(np.arange(df_snowh.values.max() - 2.1, df_snowh.values.max() - 1.0, 0.5), z)
        #z = np.append(np.arange(df_snowh.values.max() - 3.5, df_snowh.values.max() - 0.5, 0.5), z)
        z.sort()
        z_rev = z

        dt = pd.DataFrame(columns=["depths", "var"], index=df_dz.index)
        dt2 = pd.DataFrame(columns=z_rev, index=df_dz.index)

        for index, row in df_dz.iterrows(): #iterating over all of the timesteps
            z_real = df_dz.loc[index].to_list() #extract the depth
            #find index of the first element isnan
            try:
                nan_index = np.argwhere(np.isnan(z_real))[0][0]
            except IndexError:
                nan_index = -1 #this only happens when init run
            #interp wants monotonically increasing z_real and no NaNs
            z_real_rev = z_real[0:nan_index]
            z_real_rev.reverse()
            t_real_rev = df_var.loc[index][0:nan_index].to_list()
            t_real_rev.reverse() #reverse is an in place operation

            f = interpolate.interp1d(z_real_rev, t_real_rev, bounds_error=False)
            t = f(z_rev)
            dt.loc[index] = pd.Series({'depths':z_rev, 'var':t})
            for ind in range(len(t)):
                dt2.loc[index][z_rev[ind]] = t[ind]

        data=dt2["2021-12-01 03:00:00":"2021-12-15 00:00:00"]
        x_vals = np.linspace(0, len(data.index), len(data.index), dtype=int)
        y_vals = z_rev
        # y_vals = np.linspace(0, len(z), len(z), dtype=int)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        Z = data.values
        Z_list.append(Z)

    df_snowh = df_snowh.loc[hsnow.index[0]:]
    #df_snowh.index = df_snowh.index.tz_localize('UTC')
    z_005 = df_snowh - hsnow["h_TC1"] #snow height of each sensor mapped to model
    z_010 = df_snowh - hsnow["h_TC2"]
    z_020 = df_snowh - hsnow["h_TC3"]
    z_050 = df_snowh - hsnow["h_TC4"]
    z_100 = df_snowh - hsnow["h_TC5"]
    z_200 = df_snowh - hsnow["h_TC6"]
    #interpolate for each sensor height in model
    dt = pd.DataFrame(columns=["TC1", "TC2", "TC3", "TC4", "TC5", "TC6"], index=z_005.index)
    z_005 = z_005.loc[:df_snowh.index[-1]]
    for i_x in range(len(z_005)):
        # print(z_005.iloc[i]) #target to interp to
        f = interpolate.interp1d(df_dz.iloc[i_x].values, df_var.iloc[i_x].values, bounds_error=False)
        t_005 = f(z_005.iloc[i_x])
        t_010 = f(z_010.iloc[i_x])
        t_020 = f(z_020.iloc[i_x])
        t_050 = f(z_050.iloc[i_x])
        t_100 = f(z_100.iloc[i_x])
        t_200 = f(z_200.iloc[i_x])
        dt.iloc[i_x] = pd.Series({'TC1':t_005, 'TC2':t_010, 'TC3':t_020, 'TC4':t_050, 'TC5':t_100, 'TC6':t_200}, dtype=np.float64)
    dt = dt.astype(np.float64)

    ftsize=14
    fig = plt.figure(figsize=(15,10))
    axs = fig.add_subplot(1, 1, 1)
    var_dict = {
    'PSNOWTEMP': [1, cm.lajolla, 100, "Temperature", "K", [273.15]],  #[levels, cmap, nbins, label, unit]
    'PSNOWRHO': [1, cm.hawaii, 100, "Density", "kg/m3", [850]],
    'PSNOWLIQ': [1, cm.vik, 100, "Liquid content", "mmwe", [0]],
    'PSNOWHEAT': [1, cm.vik, 100, "Heat content", "J/m2", [0]],
    }
    x = Z_list[0][~pd.isnull(Z_list[0])]
    min_z = x.min()
    max_z = x.max()
    step_z = (max_z - min_z)/100.
    lev = np.arange(min_z, max_z + (2*step_z), step_z)
    cp = axs.contourf(X, Y, Z_list[0], cmap=var_dict[var_names[0]][1], norm=plt.Normalize(252., 274.), levels=lev)   #, levels=var_dict[var_names[r]][0])
    fig.colorbar(cp, ax=axs, label=f'{var_dict[var_names[0]][3]} ({var_dict[var_names[0]][4]})')
    axs.plot(X, df_snowh[:"2021-12-15 00:00:00"].values, '-k', linewidth=0.1)
    axs.set_ylim([48.0, Y.max()])

    for i in hsnow.filter(regex=r'h_').columns:
        i_c = i.split('_')[1]
        df_test = c[[f'{i_c}_Avg', 'depth']]
        df_test["depth"] = (hsnow_m["SNOWH"][0] - hsnow[i])
        df_test[f'{i_c}_Avg'] = df_test[f'{i_c}_Avg']+273.15
        df_test[f'{i_c}_diff'] = dt[f'{i_c}'] - df_test[f'{i_c}_Avg']
        x_c = x_vals
        y_c=df_test["depth"].values
        col=df_test[f'{i_c}_diff'].values
        #plots every 5th dot
        diff = axs.scatter(x_c[::5], y_c[::5], c=col[::5], cmap='seismic', marker='o', norm=plt.Normalize(-10., 10.), edgecolors='white', linewidth=0.2) #, marker='o', edgecolors='grey') #, levels=lev)
        #plt.colorbar()
    cbb = plt.colorbar(diff)
    my_xticks = []
    for i in data.index.values:
        my_xticks.append(i)
    my_xticks2 = [re.sub(r'\w\d\d\:00\:00\.0+$', '', str(d)) for d in my_xticks]

    cbb.set_label('modelled - observed (K)')
    plt.xticks(list(range(0,len(data.index),1)), my_xticks2, rotation=45)
    n=var_dict["PSNOWTEMP"][2]
    plt.locator_params(axis='x', nbins=len(X)/24.)
    plt.ylim(48.9, 50.1)
    plt.subplots_adjust(top=0.934, bottom=0.145, left=0.063, right=0.99, hspace=0.19, wspace=0.2)
    fig.supylabel('Height (m)', fontsize=12)
    #plt.show()
    plt.savefig(f'{save_dir}/xsecticetempdiff_{station_name}.png', bbox_inches='tight')

def plot_xsecticetempspinnup(newrunoff_newalbedo, c, save_dir, station_name):

    """
    function to plot cross section of modelled glacier temps over full model period from spinnup to evaluation period

    """
    hsnow_m = pd.DataFrame()
    hsnow_m["SNOWH"] = newrunoff_newalbedo["SNOWH"].loc["2021-08-01 14:00:00+00:00":]
    #hsnow_m["SNOWH"] = df["SNOWH"].loc["2021-12-01 13:00:00":"2021-12-15 00:00:00"]
    hsnow_m["SNOWH_scaled"] = hsnow_m["SNOWH"] - hsnow_m["SNOWH"][0]

    var_names = ["PSNOWTEMP", "PSNOWRHO", "PSNOWLIQ", "PSNOWHEAT"]
    Z_list = []
    cp_list = []

    for var_name in var_names[:1]:
        df_snowh, df_dz, df_var = prep.proc_xsection(save_dir+'/newrunoff_newalb/NWM', station_name, var_name)
        z = np.arange(df_snowh.values.max() - 0.7, df_snowh.values.max(), 0.005)
        z = np.append(np.arange(df_snowh.values.max() - 3.5, df_snowh.values.max() - 0.6, 0.01), z)
        z.sort()
        z_rev = z

        dt = pd.DataFrame(columns=["depths", "var"], index=df_dz.index)
        dt2 = pd.DataFrame(columns=z_rev, index=df_dz.index)

        for index, row in df_dz.iterrows(): #iterating over all of the timesteps
            z_real = df_dz.loc[index].to_list() #extract the depth
            #find index of the first element isnan
            try:
                nan_index = np.argwhere(np.isnan(z_real))[0][0]
            except IndexError:
                nan_index = -1 #this only happens when init run
            #interp wants monotonically increasing z_real and no NaNs
            z_real_rev = z_real[0:nan_index]
            z_real_rev.reverse()
            t_real_rev = df_var.loc[index][0:nan_index].to_list()
            t_real_rev.reverse() #reverse is an in place operation

            f = interpolate.interp1d(z_real_rev, t_real_rev, bounds_error=False)
            t = f(z_rev)
            dt.loc[index] = pd.Series({'depths':z_rev, 'var':t})
            for ind in range(len(t)):
                dt2.loc[index][z_rev[ind]] = t[ind]

        data=dt2
        x_vals = np.linspace(0, len(data.index), len(data.index), dtype=int)
        y_vals = z_rev
        # y_vals = np.linspace(0, len(z), len(z), dtype=int)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        Z = data.values
        Z_list.append(Z)

    ftsize=14
    fig = plt.figure(figsize=(15,10))
    axs = fig.add_subplot(1, 1, 1)
    var_dict = {
    'PSNOWTEMP': [1, cm.oslo, 100, "Temperature", "K", [273.15]],  #[levels, cmap, nbins, label, unit]
    'PSNOWRHO': [1, cm.hawaii, 100, "Density", "kg/m3", [850]],
    'PSNOWLIQ': [1, cm.vik, 100, "Liquid content", "mmwe", [0]],
    'PSNOWHEAT': [1, cm.vik, 100, "Heat content", "J/m2", [0]],
    }
    x = Z_list[0][~pd.isnull(Z_list[0])]
    min_z = x.min()
    max_z = x.max()
    step_z = (max_z - min_z)/100.
    lev = np.arange(min_z, max_z + (2*step_z), step_z)
    cp = axs.contourf(X, Y, Z_list[0], cmap=var_dict[var_names[0]][1], norm=plt.Normalize(min_z, 274.), levels=lev)
    fig.colorbar(cp, ax=axs, label=f'{var_dict[var_names[0]][3]} ({var_dict[var_names[0]][4]})')
    axs.plot(X, df_snowh.values, '-k', linewidth=0.1)

    my_xticks = []
    for i in data.index.values:
        my_xticks.append(i)
    my_xticks2 = [re.sub(r'-\d\d\w\d\d\:00\:00\.0+$', '', str(d)) for d in my_xticks]

    plt.xticks(list(range(0,len(data.index),1)), my_xticks2, rotation=45)
    n=var_dict["PSNOWTEMP"][2]
    plt.locator_params(axis='x', nbins=len(X)/744.)
    plt.ylim(47.0, 50.5)
    plt.subplots_adjust(top=0.934, bottom=0.145, left=0.063, right=0.99, hspace=0.19, wspace=0.2)
    fig.supylabel('Height (m)', fontsize=12)
    plt.savefig(f'{save_dir}/xsecticetempspinnup_{station_name}.png', bbox_inches='tight')
