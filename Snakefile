import glob
import re

configfile: "config.yaml"

FILE_DIR = config['file_dir']
# directory to save plots to
SAVE_DIR = config['save_dir']
STATION_NAMES = config['station_names'] 
PLOT_NAMES = config['plot_names']

rule all:
    input:
        expand(f'{SAVE_DIR}/{pname}_{STATION_NAMES}.png' for pname in PLOT_NAMES),

#rule clean:
#    shell:
#        "rm {SAVE_DIR}/*.png"

rule createTimeseriesPlot:
    input:
        "{FILE_DIR}/timeseries_ldasout_{STATION_NAMES}.csv"
    output:
        report("{SAVE_DIR}/{pname}_{STATION_NAMES}.png", category="plot")
    shell:
        "python main_plotpaper1.py --save-dir={SAVE_DIR} --station-name={STATION_NAMES} --plot-name={wildcards.pname}"

