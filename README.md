## Dependencies
To setup the dependencies, simply run:
```
pip install -r requirements.txt
```
You can also conda install the requirements you need from requirements.txt if you choose.

## Run the Algorithms & Final Results
The experiment files (Weighbor_experiment.py, FastNJ_experiment.py, ...) are self-containing and hold all necessary functions and data to run a simulated experiment for that specific algorithm.
Specifically, the functions include simulating a simplified birth-death tree, simulating sequences under Jukes-Cantor, computing pairwise distances, the implementation of the NJ-variant, computing Robinson-Foulds (RF) distance, assembling the tree and generating the Newick representation, and running the experiment for multiple dataset sizes.
All you need to do is run the program, which will run the main function. This will ultimately print out the average runtime and its st. dev across all runs, as well as the average RF distance and its st. dev across all runs.

## Using Real Data
