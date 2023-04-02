# wrf_hydro_nwm_coldglacier_postprocessworkflow
Postprocessing workflow for the wrf-hydro runs and figures in "The application and modification of WRF-Hydro-Crocus to a cold-based Antarctic glacier"
## Requirements

You will need to have "Snakemake" installed. The easiest is to create a conda environment
```
conda create -p /path/to/snakemake_env
conda activate /path/to/snakemake_env
conda install -c conda-forge -c bioconda snakemake
```

## How to check the syntax of your workflow

```
snakemake --dry-run
```

## How to run the workflow

Using 4 workers (for instance)
```
snakemake -j 4
```

The `output` directory will coontain the output files, including the intermediate files:
```
$ ls output/
albedoafter_cwg.png   forcing_cwg.png   sfctempscatter_cwg.png  xsecticetempdiff_cwg.png
albedobands_cwg.png   icetempH_cwg.png  runoff3plot_cwg.png      sfctemptime_cwg.png           
xsecticetempspinnup_cwg.png  albedobefore_cwg.png  runoffcomp_cwg.png snowheight_cwg.png runoffmod_cwg.png   
xsecticetemp_cwg.png
```

## How to clean up the output directory

```
snakemake -j 1 clean
```

## How to create a report

Once you have run your workflow,
```
snakemake --report report.html
```

## How to submit yyour workflow to mahuika

Edit the `mahuika.json` file. Then type
```
snakemake -j 999 --cluster-config mahuika.json --cluster "sbatch --account={cluster.account} --ntasks={cluster.ntasks} --time={cluster.time}"
```
