# Dataset Managers
For enviorment setup, please refer to the [main README](../README.md).

> Note that `pairwise_generation_dm.py` contains the original code for the paper so, `dialog_dm.py` and `spoken_dm.py` were produced for QoL changes and contains the same daat preprocessing required

Contains the dataset managers for the various datasets used in this project.
The `dialog_dm.py` contains the main dataset manager for all of the datasets which includes written and spontaneous dialog datasets.

## `spoken_dm.py`
Contains the dataset manager for the spoken dialog datasets. This includes the Switchboard, Fisher and Edinburgh Accents datasets.
This data was accessed under different licenses so, we do not make this data publically available.

To examine the output of the pre-processing pipeline run the following:
```bash
python spoken_dm.py --dataset switchboard --split train --show-output --process "aligned"
```

Other arguments are available to specify the dataset, split and pre-processing pipeline.

### Examples
1. Time Aligned (w/o backchannels and w/o overlaps)
```bash
python spoken_dm.py --dataset switchboard --split train --show-output --process "aligned"
```
2. Time Aligned With Backchannels (w/o overlaps)
```bash
python spoken_dm.py --dataset switchboard --split train --show-output  --process "aligned" --include-backchannels
```
3. Time Aligned With Backchannels and With Overlaps
```bash
bash spoken_dm.py --dataset switchboard --split train --show-output --process "aligned" --include-backchannels --include-overlaps
```
4. Time Aligned With Backchannels and With Overlaps (With Yield Tokens)
```bash
python spoken_dm.py --dataset switchboard --split train --show-output --process "aligned" --include-backchannels --include-overlaps --include-yield-token
```
5. Serialised
```bash
python spoken_dm.py --dataset switchboard --split train --show-output --process "serialised"
```
6. Serialised (With Yield Tokens)
```bash
python spoken_dm.py --dataset switchboard --split train --show-output --process "serialised" --include-yield-token
```
7. Serialised and Combined Speaker Channels (Output is a single stream)
```bash
python spoken_dm.py --dataset switchboard --split train --show-output --process "serialised" --combine-speaker
```

### Processors
Processors are used to pre-process the data in different ways.
The `aligned` processor forces turns to be time aligned so that the start and end of turns can be overlapped.
As a result, we can include backchannels and complete overlapps that have to be removed for other processors.
Additionally, the `aligned` processor can include yield tokens to indicate when we have determined that a speaker has ended their turn from the influence of the other speaker.

The `serialised` processor serialises the data so that the start and end of turns are not overlapped.
Optionally, the `serialised` processor can include yield tokens and turn types that are based on the original position of the turn in the conversation.
Also, the two channels can be combined into a single channel via `combine_speaker'

### Dataset Specific
Each dataset has its own pre-processing pipeline that is used to convert the raw data into a format that can be processed later.
This involves assigning speaker roles, removing unwanted tokens and assigning utterance and word-level timings.
These timings allow us to distinguish and divide the utterances into their main turns, backchannels and overlaps.


## `written_dm.py`
Contains the dataset manager for the written dialog datasets.
