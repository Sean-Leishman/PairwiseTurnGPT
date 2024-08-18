# Dataset Managers

### Switchboard
Switchboard is the main dataset used in this project. It is a collection of telephone conversations between two speakers.
We use a licensed version of the Switchboard dataset which is available from LDC.
For our purposes, we provide the dataset via Git LFS to ensure the data is of a small size and can be easily downloaded.
After cloning the repository you can download the data by running `git lfs fetch --all` and running `bash data/switchboard/extract_swb.sh` to extract the data.

## PairwiseGenerationDM
The main source of data for PairwiseTurnGPT. 
The original data is obtained, like in TurnGPT from the Switchboard folder 
which extracts the list of dialogs and additionally if required the backchannels
and overlaps.

From this we can tag, on the turn-level, whether a turn interrupted the previous turn, is a backchannel, an overlap or none of the above (NORMAL)
This is used to label the end of turns as either NORMAL or YIELD as an end-of-turn is a YIELD if it is interrupted by the other speaker's turn and they continue speaking or the other speaker had an overlap close to turn-end. 

By doing so, each token within a turn has some turn level information associated with it.
We align tokens from each turn purely based on their timings and if they overlap, they are inserted simultaneously.
Otherwise, a corresponding `<emp>` token is added in place for a token.
Add turn-ends an additional token is added to indicate the end of the turn.

For the serialised versions, we want to have the yield turn endings of the aligned version.
Therefore we always calculate the turn-level information but when inserted tokens we ignore their timing information and instead insert all the tokens in a turn without considering overlap. 
This can then be used to produce a single stream by simply not inserting, into a new stream, the pair `<emp>` token.

You can find parameter settings from `PairwiseGenerationDM` class but note, if `basic_mode` is set then we automatically `remove_overlaps`, `remove_backchannels` and `remove_start_tokens`. 
If `combine_speaker` is set then we automatically set `basic_mode` to True.

### Examples
The examples below use a key `sw4617A-ms98-a-0001` to pick a specific dialog from the dataset to tokenize via `parse_dialogs`.
Removing the `parse_dialogs` argument will cause the entire split to be tokenized (split="all"/"train"/"test"/"val"). 
The Pytorch dataset is saved in the `data/switchboard/.cache` folder and can subsequently be loaded by setting `overwrite=False` and we should be able to interact with a previously generated dataset.

1. Time Aligned With Overlaps & Backchannels (With Yield Tokens)
```python
conda activate ENV
python 

> from data.pairwise_generation_dm import PairwiseGenerationDM
> tokenizer = SpokenDialogTokenizer(tokens=[
    '<bc>', '<yield>', '<emp>', '<eot>', '<eint>', '<ebc>', '<speakerA>', '<speakerB>'
 ])
> ts = PairwiseGenerationDM(tokenizer=tokenizer, split="test", overwrite=True, remove_start_tokens=True,
                      include_yield_token=True,
                      combine_speaker=False, basic_mode=False, datasets=['switchboard'],
                      dev_mode=False, remove_overlaps=False, remove_backchannels=False, no_emp_tokens=False, savedata=False, include_overlap_token=True,
                      include_bc_token=False, include_end_bc_token=True, individual_speaker_tokens=False, filter_bc_overlap_token=False,
                      parse_dialogs=["4617"]
                      )
> ts.prepare_data()
> ts.pp_item(f'sw4617A-ms98-a-0001')
```

2. Time Aligned With Overlaps (With Yield Tokens)
```python
conda activate ENV
python 

> from data.pairwise_generation_dm import PairwiseGenerationDM
> tokenizer = SpokenDialogTokenizer(tokens=[
    '<bc>', '<yield>', '<emp>', '<eot>', '<eint>', '<ebc>', '<speakerA>', '<speakerB>'
 ])
> ts = PairwiseGenerationDM(tokenizer=tokenizer, split="test", overwrite=True, remove_start_tokens=True,
                      include_yield_token=True,
                      combine_speaker=False, basic_mode=False, datasets=['switchboard'],
                      dev_mode=False, remove_overlaps=False, remove_backchannels=True, no_emp_tokens=False, savedata=False, include_overlap_token=True,
                      include_bc_token=False, include_end_bc_token=True, individual_speaker_tokens=False, filter_bc_overlap_token=False,
                      parse_dialogs=["4617"]
                      )
> ts.prepare_data()
> ts.pp_item(f'sw4617A-ms98-a-0001')
```

3. Serialised With Dual Streams (With Yield Tokens)
```python
conda activate ENV
python 

> from data.pairwise_generation_dm import PairwiseGenerationDM
> tokenizer = SpokenDialogTokenizer(tokens=[
    '<bc>', '<yield>', '<emp>', '<eot>', '<eint>', '<ebc>', '<speakerA>', '<speakerB>'
 ])
> ts = PairwiseGenerationDM(tokenizer=tokenizer, split="test", overwrite=True, remove_start_tokens=True,
                      include_yield_token=True,
                      combine_speaker=False, basic_mode=True, datasets=['switchboard'],
                      dev_mode=False,no_emp_tokens=False, savedata=False, include_overlap_token=True,
                      include_bc_token=False, include_end_bc_token=True, individual_speaker_tokens=False, filter_bc_overlap_token=False,
                      parse_dialogs=["4617"]
                      )
> ts.prepare_data()
> ts.pp_item(f'sw4617A-ms98-a-0001')
```

4. Serialised With Single Stream (With Yield Tokens)
```python
conda activate ENV
python 

> from data.pairwise_generation_dm import PairwiseGenerationDM
> tokenizer = SpokenDialogTokenizer(tokens=[
    '<bc>', '<yield>', '<emp>', '<eot>', '<eint>', '<ebc>', '<speakerA>', '<speakerB>'
 ])
> ts = PairwiseGenerationDM(tokenizer=tokenizer, split="test", overwrite=True, remove_start_tokens=True,
                      include_yield_token=True,
                      combine_speaker=True, basic_mode=True, datasets=['switchboard'],
                      dev_mode=False,no_emp_tokens=False, savedata=False, include_overlap_token=True,
                      include_bc_token=False, include_end_bc_token=True, individual_speaker_tokens=False, filter_bc_overlap_token=False,
                      parse_dialogs=["4617"]
                      )
> ts.prepare_data()
> ts.pp_item(f'sw4617A-ms98-a-0001')
```

4. Single Stream 
```python
conda activate ENV
python 

> from data.pairwise_generation_dm import PairwiseGenerationDM
> tokenizer = SpokenDialogTokenizer(tokens=[
    '<bc>', '<yield>', '<emp>', '<eot>', '<eint>', '<ebc>', '<speakerA>', '<speakerB>'
 ])
> ts = PairwiseGenerationDM(tokenizer=tokenizer, split="test", overwrite=True, remove_start_tokens=True,
                      include_yield_token=True,
                      combine_speaker=True, basic_mode=True, datasets=['switchboard'],
                      dev_mode=False,no_emp_tokens=True, savedata=False, include_overlap_token=True,
                      include_bc_token=False, include_end_bc_token=True, individual_speaker_tokens=False, filter_bc_overlap_token=False,
                      parse_dialogs=["4617"]
                      )
> ts.prepare_data()
> ts.pp_item(f'sw4617A-ms98-a-0001')
```
