# PairwiseTurnGPT

This repository contains the code for PairwiseTurnGPT, an End-of-turn detection model trained on the Switchboard corpus
PairwiseTurnGPT is based on the work of [TurnGPT](https://github.com/ErikEkstedt/TurnGPT) but modifies the model and spoken dialogue processing to effectively model speaker interactions in overlapping utterances.

This allows us to utilize phenomena such as backchannels, interruptions and overlaps in predicting end-of-turns in spoken dialogues without the need for acoustic features. 

The model itself is based on the GPT-2 architecture but uses a dual-transformer approach to model each speaker's utterances separately. 

For easy comparison, the original TurnGPT model has been cloned in this repository within the `gptonly` folder.
The `pairwisegpt` folder contains the code for PairwiseTurnGPT.

## Setup
Ran and tested on a ubuntu-based linux machine with conda.

A GPU supported machine is required for training and testing.

```
conda create -n ENV_NAME python=3 --file package_list.txt
conda activate ENV_NAME
pip install -r requirements.txt
```
```
cd data && pip install -e .
```
```
pip install -e .
```

## File Structure
There are two main folders `gptonly`, for TurnGPT, and `pairwisegpt` for PairwiseTurnGPT. 
Here follows the contents of said folders with reference to where code has not been 
modified for this project.

Each contains:
- tokenizer.py: written for the original [TurnGPT](https://github.com/ErikEkstedt/TurnGPT)
- train.py: contains entry code
- trainer.py: contains main train and testing loops
- model.py: contains model intialisation and tokenization initialisation
- gpt.py: contains model architecture modified from original [Hugging Face](https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/gpt2/modeling_gpt2.py) version
- metrics.py: contains classes for tracking metrics
- evaluate.py: miscellaneous code for experiments related to generation and artifical removals of overlaps
- generate.py: written for the original [TurnGPT](https://github.com/ErikEkstedt/TurnGPT), adapted for dual stream tokens
- utils.py: miscellaneous code with functions from original [TurnGPT](https://github.com/ErikEkstedt/TurnGPT)
- config.py: contains runs that is used for training runs 

The data folder contains the necessary code for loading and running functions
- pairwise_generation_dm.py: main code for parsing and tokenizing all conversational data
- switchboard/switchboard.py: adapted from original [TurnGPT](https://github.com/ErikEkstedt/TurnGPT) calling additional functions for pairwise approach
- switchboard/utils.py: adapted from original [TurnGPT](https://github.com/ErikEkstedt/TurnGPT) with pairiwse setup and retaining phenomna

## Data
Switchboard data is aquired from [https://www.inf.ed.ac.uk/resources/corpora/](https://www.inf.ed.ac.uk/resources/corpora/) for Switchboard. 
But alternatively a download script is provided in [data/switchboard/extract_swb.bash](data/switchboard/extract_swb.bash) to download the data.
More information can be found at `data/README.md` and `data/switchboard/README.md`.
Interact with the data subfolder directly to understand how the data is loaded and tokenized.


## Training Runs
To acquire the runs as described in the paper look through `config.py` for each folder.
Generally this is run using:

### TurnGPT Run Options
```bash
python gptonly/train.py --result-run RUN_NAME
```
- `experiment1`

### PairwiseTurnGPT Run Options
```bash
python pairwisegpt/train.py --result-run RUN_NAME
```
- `experiment1`
- `experiment2`
- `experiment3`

## Run Options
Optionally, custom runs can be achieved by setting appropriate arguments to `train.py` which can be accessed via `python gptonly/train.py --help` or `python pairwisegpt/train.py --help`.

