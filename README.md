# active-learning-segmentation-baselines

PyTorch implementations of active learning heuristics for deep semantic segmentation on histology image data. 
The code in this directory focuses on [deep bayesian active learning](https://arxiv.org/abs/1703.02910) techniques. 
Monte Carlo dropout is used to approximate a model's epistemic uncertainty in order to draw samples from a
pool for annotation, which aims to minimize the labeling cost and maximise model performance.

For an overview of active learning in general see this [survey](http://burrsettles.com/pub/settles.activelearning.pdf).



## Active learning heuristics

The AL heuristics found in the following papers were implemented and adapted for semantic segmentation.

1. [Deep Active Learning for Axon-Myelin Segmentation on Histology Data](https://arxiv.org/abs/1907.05143).

2. [BALD](https://arxiv.org/abs/1703.02910)

3. [Max entropy](https://arxiv.org/abs/1703.02910)

## Requirements

Install the requirements with pip ```$ pip install -r requirements.txt```

## Dataset

In order to use the GlaS histology dataset, first download it from the [site](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/).
Place the extracted files in the data directory using this structure :

```
.
├── data
│   ├── GlaS
│   │   ├── Grade.csv
│   │   ├── testA_1.bmp
│   │   ├── testA_1_anno.bmp
│   │   ├── testA_2.bmp
│   │   └── ...
...
```

Then run the ``` data/create_splits.py ``` script. This will create the train, test and validation splits 
for the GlaS dataset under the ```data/splits/glas/``` directory.

## Experiment code

The only segmentation model in the repository right now is [UNet](https://arxiv.org/abs/1505.04597). 
To run an experiment, you can use the [Sacred package](https://pypi.org/project/sacred/) command line interface. 
To run 5 active learning experiments with BALD using 20 MC dropout iterations : 

```
python3 active.py with heuristic='bald' run=0 mc_iters=20 results_dir='results'
python3 active.py with heuristic='bald' run=1 mc_iters=20 results_dir='results'
python3 active.py with heuristic='bald' run=2 mc_iters=20 results_dir='results'
python3 active.py with heuristic='bald' run=3 mc_iters=20 results_dir='results'
python3 active.py with heuristic='bald' run=4 mc_iters=20 results_dir='results'
```

Where the ```run``` number corresponds to an experiment's number, mean dice scores over the test set are saved as numpy 
arrays in the results_dir. Other default experiment parameters can be changed using the cmd line interface, to see them
 go to the ``` conf()``` Sacred decorator function in [active.py](active.py).
 
 ## Acknowledments

This repository uses code from [survey_wsl_histology](https://github.com/jeromerony/survey_wsl_histology/blob/master/README.md) 
for the UNet model and data processing for GlaS, and from [baal](https://baal.readthedocs.io/en/latest/) for the useful 
active learning dataset wrapper and MC dropout patcher.