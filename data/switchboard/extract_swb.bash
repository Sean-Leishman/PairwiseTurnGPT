#!/bin/bash

curl https://isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz -o switchboard.tar.gz;
tar -xvzf switchboard.tar.gz;
mv swb_ms98_transcriptions switchboard/switchboard1/transcriptions;
