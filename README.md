## Dependencies
To setup the dependencies, simply run:
```
pip install -r requirements.txt
```
You can also conda install the requirements you need from requirements.txt if you choose.

## Synthetic Benchmark

The synthetic experiment scripts are in folder `benchmark_scripts`. The experiment files (Weighbor_experiment.py, FastNJ_experiment.py, ...) are self-containing and hold all necessary functions and data to run a simulated experiment for that specific algorithm. Simply execute a script (no arguments) to run an experiment. Performance metrics will be printed to standard out.

The experiment outputs from our testing can be found in `synth_data`.

The plots from our experimental data as well as the script generating the plots can be found in folder `benchmark_figures`.

## Using Real Data

See folder `TreeBASE` for extracting sequence and tree data from TreeBASE. Inside includes a README detailing the process as well as the specific TreeBASE resources we used.

See folder `RealData` for how we conducted analysis on our extracted data from TreeBASE. Inside includes a README for how to run the scripts.
