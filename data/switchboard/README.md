# Switchboard Act Corpus

## Transcriptions
For transcriptions, the corpus is split into two main parts. `cellular` and `switchboard` which will be used as a verified portion of transcriptual data. 

A conversation is recorded and transcribed as a full conversation where each record is a turn or on word by word basis where each word is on a separate line. This is done as word alignment is performed such that the timing of each word is recorded as well as the timing of the entire utterance. 

The data is currently in the format where each speaker has their own transcript and word files with time alignments. As such there are no clear speaker turns available in the data and as such the data has to be augmented to provide this.

As well as this there are various other features or tokens added to the dataset which are not relevant in other datasets and as such should be removed. This step of preprocessing data was taken from [TurnGPT](https://github.com/ErikEkstedt/datasets_turntaking/blob/main/datasets_turntaking/dataset/switchboard/README.md)
