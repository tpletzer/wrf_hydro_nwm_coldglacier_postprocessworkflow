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
import datetime

def get_diam(PSNOWGRAN1, PSNOWGRAN2):

    """
    PARAMETERS
    """
    XD1 = 1.
    XD2 = 3.
    XD3 = 4.
    XX = 99.

    if PSNOWGRAN1<0.:
        PDIAM = -PSNOWGRAN1*XD1/XX + (1.+PSNOWGRAN1/XX)*(PSNOWGRAN2*XD2/XX + (1.-PSNOWGRAN2/XX) * XD3) 
        PDIAM = PDIAM/10000.      

    else:
        PDIAM = PSNOWGRAN2*PSNOWGRAN1/XX + max(0.0004, 0.5*PSNOWGRAN2 )*(1.-PSNOWGRAN1/XX)      

    return PDIAM

def get_alb(PSNOWRHO_IN,PPS_IN,PVAGE1,PSNOWGRAN1,PSNOWGRAN2,PSNOWAGE, XVALB, XVDIOP1, XVRPRE1, XVRPRE2, XVPRES1, model):

    """
    PARAMETERS
    """

    # XVALB2 = .96
    # XVALB3 = 1.58
    # XVALB4 = .92
    # XVALB5 = .90
    # XVALB6 = 15.4
    # XVALB7 = 346.3
    # XVALB8 = 32.31
    # XVALB9 = .88
    # XVALB10 = .200
    # XVALB11 = .65

    # XVDIOP1 = 2.3e-3
    # XVRPRE1 = 1.5 #try bigger 1.095769845545977 than og:0.5 
    # XVRPRE2=1.5
    # XVPRES1 = 87000.
    PALB = []
    ZDIAM = get_diam(PSNOWGRAN1,PSNOWGRAN2)
    if PSNOWRHO_IN < 850.:

        # ZDIAM = get_diam(PSNOWGRAN1,PSNOWGRAN2)

        ZDIAM_SQRT = np.sqrt(ZDIAM)

        PALB = []
        PALB.insert(0, min(XVALB[0] - XVALB[1]*ZDIAM_SQRT, XVALB[2]))
        PALB.insert(1, max(XVALB[10], XVALB[3] - XVALB[4]*ZDIAM_SQRT)) ####CHANGE 0.3!!!!
        ZDIAM   = min( ZDIAM, XVDIOP1 )
        ZDIAM_SQRT = np.sqrt(ZDIAM)
        PALB.insert(2, max(0., XVALB[5]*ZDIAM - XVALB[6]*ZDIAM_SQRT + XVALB[7]))
        #PALB.insert(0, max(XVALB[9],PALB[0] - min(max(PPS_IN/XVPRES1,XVRPRE1), XVRPRE2)*XVALB[8] * PSNOWAGE / PVAGE1)) 
        PALB[0] = max(XVALB[9],PALB[0] - min(max(PPS_IN/XVPRES1,XVRPRE1), XVRPRE2)*XVALB[8] * PSNOWAGE / PVAGE1)

    else:
        if model=='oldalbedo':
            PALB.insert(0, 0.23) 
            PALB.insert(1, 0.15)
            PALB.insert(2, 0.06)
        else:
            PALB.insert(0, 0.7) 
            PALB.insert(1, 0.3)
            PALB.insert(2, 0.05)

    return PALB, ZDIAM


def snowcroalb(PSNOWDZ,PSNOWRHO,PSNOWGRAN1_TOP,PSNOWGRAN2_TOP,PSNOWAGE_TOP,PSNOWGRAN1_BOT,
                PSNOWGRAN2_BOT,PSNOWAGE_BOT,PPS, 
                XVALB, XVDIOP1, XVRPRE1, XVRPRE2, XVPRES1, model):

    """
    PARAMETERS
    """

    # XANSMIN=0.50 
    # XANSMAX =0.85 #max snow albedo
    XVAGING_NOGLACIER = 60.
    # XVAGING_GLACIER  = 900. 
    # XVPRES1 = 87000.
    XVSPEC1 = 0.71
    XVSPEC2 = 0.21
    XVSPEC3 = 0.08

    # ZANSMIN = XANSMIN
    # ZANSMAX = XANSMAX
    ZVAGE1  = XVAGING_NOGLACIER

    XVW1 = .80
    XVW2 = .20
    XVD1 = .02
    XVD2 = .01


    ZALB_TOP,ZDIAM_TOP = get_alb(PSNOWRHO[0],PPS,ZVAGE1,PSNOWGRAN1_TOP,PSNOWGRAN2_TOP,PSNOWAGE_TOP, XVALB, XVDIOP1, XVRPRE1, XVRPRE2, XVPRES1, model)
    ZALB_BOT,ZDIAM_BOT = get_alb(PSNOWRHO[1],PPS,ZVAGE1,PSNOWGRAN1_BOT,PSNOWGRAN2_BOT,min(365.,PSNOWAGE_BOT), XVALB, XVDIOP1, XVRPRE1, XVRPRE2, XVPRES1, model)

    ZMIN = min(1., PSNOWDZ/XVD1)
    ZMAX = max(0., (PSNOWDZ-XVD1)/XVD2)
    ZFAC_TOP = XVW1 * ZMIN + XVW2 * min(1., ZMAX)
    ZFAC_BOT = XVW1 * (1.-ZMIN) + XVW2 * (1.-min(1., ZMAX))
    # ZFAC_BOT = 0.
    # ZFAC_TOP = 1. 

    PSPECTRALALBEDO = []
    for lev in range(0, 3):
        PSPECTRALALBEDO.insert(lev, ZFAC_TOP * ZALB_TOP[lev] + ZFAC_BOT * ZALB_BOT[lev])

    XVSPEC = [XVSPEC1, XVSPEC2, XVSPEC3]
    PALBEDOSC = sum([i*j for (i, j) in zip(PSPECTRALALBEDO, XVSPEC)])

    return PALBEDOSC, PSPECTRALALBEDO, ZALB_TOP, ZALB_BOT,ZDIAM_TOP,ZDIAM_BOT



def snowcroalb_3param(p_dict, df, model):
    palb_dict = {}
    palb_band_dict = {}
    zalb_top_dict = {}
    zalb_bot_dict = {}
    zdiam_dict = {}

    for p in range(3):

        PSNOWALB = []
        BAND1 = []
        BAND2 = []
        BAND3 = []

        BAND1_top = []
        BAND2_top = []
        BAND3_top = []
        BAND1_bot = []
        BAND2_bot = []
        BAND3_bot = []

        ZDIAM_TOP = []
        ZDIAM_BOT = []
        xvalb = p_dict[p]
        # xvalb12 = p #og 0.3
        for i in df.index:
            alb_calc = df.loc[i]

            PSNOWDZ0 = alb_calc["PSNOWDZ0"]
            PSNOWRHO = alb_calc["PSNOWRHO01"]
            PSNOWGRAN1_0 = alb_calc["PSNOWGRAN1_0"]
            PSNOWGRAN2_0 = alb_calc["PSNOWGRAN2_0"]
            PSNOWGRAN1_1 = alb_calc["PSNOWGRAN1_1"]
            PSNOWGRAN2_1 = alb_calc["PSNOWGRAN2_1"]
            PSNOWAGE0 = alb_calc["PSNOWAGE0"]
            PSNOWAGE1 = alb_calc["PSNOWAGE1"]
            P = alb_calc["P"]

            alb, spectral, zalb_top, zalb_bot, zdiam_top, zdiam_bot = snowcroalb(PSNOWDZ0,PSNOWRHO, PSNOWGRAN1_0,PSNOWGRAN2_0,PSNOWAGE0,PSNOWGRAN1_1,PSNOWGRAN2_1,PSNOWAGE1,P,
                            xvalb, 2.3e-3, 1.5, 1.5, 87000., model)
            PSNOWALB.append(alb)
            BAND1.append(spectral[0])
            BAND2.append(spectral[1])
            BAND3.append(spectral[2])

            BAND1_top.append(zalb_top[0])
            BAND2_top.append(zalb_top[1])
            BAND3_top.append(zalb_top[2])

            BAND1_bot.append(zalb_bot[0])
            BAND2_bot.append(zalb_bot[1])
            BAND3_bot.append(zalb_bot[2])

            ZDIAM_TOP.append(zdiam_top)
            ZDIAM_BOT.append(zdiam_bot)

        palb_dict[np.round(p,3)] =  PSNOWALB
        palb_band_dict[f'{np.round(p,3)}_1'] =  BAND1
        palb_band_dict[f'{np.round(p,3)}_2'] =  BAND2
        palb_band_dict[f'{np.round(p,3)}_3'] =  BAND3

        zalb_top_dict[f'{np.round(p,3)}_1'] = BAND1_top
        zalb_top_dict[f'{np.round(p,3)}_2'] = BAND2_top
        zalb_top_dict[f'{np.round(p,3)}_3'] = BAND3_top

        zalb_bot_dict[f'{np.round(p,3)}_1'] = BAND1_bot
        zalb_bot_dict[f'{np.round(p,3)}_2'] = BAND2_bot
        zalb_bot_dict[f'{np.round(p,3)}_3'] = BAND3_bot

        zdiam_dict[f'{np.round(p,3)}_diam_top'] = ZDIAM_TOP
        zdiam_dict[f'{np.round(p,3)}_diam_bot'] = ZDIAM_BOT

    palb_df = pd.DataFrame.from_dict(palb_dict)
    palb_df.index = df.index

    palb_band_df = pd.DataFrame.from_dict(palb_band_dict)
    palb_band_df.index = df.index

    zalb_top_df = pd.DataFrame.from_dict(zalb_top_dict)
    zalb_top_df.index = df.index
    #zalb_top_df.plot(), plt.show()

    zalb_bot_df = pd.DataFrame.from_dict(zalb_bot_dict)
    zalb_bot_df.index = df.index

    zdiam_df = pd.DataFrame.from_dict(zdiam_dict)
    zdiam_df.index = df.index

    return palb_df, palb_band_df, zdiam_df, zalb_top_df, zalb_bot_df
