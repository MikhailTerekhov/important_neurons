# Are important neurons more interpretable?
This repository provides supplementary code for my final project for the [AGI Safety Fundamentals](https://course.agisf.com/ai-alignment) course.

## Requirements
The project requires TensorFlow 2 and several other standard packages that can be installed with `pip install -r requirements.txt`. The only nontrivial dependence is [my fork of Lucid](https://github.com/MikhailTerekhov/lucid), which should in principle also get installed by the above command.

## Provided scripts
- `run_experiment.py` generates the visualizations and comuputes importance of the features in the specified network and its layer. Under the hood it uses `model_wrapper.py` that provides convenience methods for Lucid's models.
- `select_by_importance.py` can be run on the data generated by `run_experiment` to select the groups of features whose importance needs to be evaluated.
- `run_survey.py` shows the survey GUI and collects responses. The neurons selected on the previous step are hard-coded in the script.
- `analyze.py` takes in the data from both `run_experiment` and `run_survey` and produces the plot used in the report.

## Results
I've run the experiment on the layer `mixed5a` of the InceptionV1 net. My importance measurements, vizualizations, and survey answers are stored in `data/neurons_complete` folder. The report with the analysis of the results is available [here](`report/report.pdf`).