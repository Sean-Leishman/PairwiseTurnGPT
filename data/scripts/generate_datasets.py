from data.spoken_dm import SpokenDM
from data.base import Datasets

dataset = [Datasets.SWITCHBOARD]


for split_utt in [True, False]:
    for combine_speaker in [True, False]:
        spoken_dm = SpokenDM(
            datasets=dataset,
            process="serialised",
            serialised="turn-end",
            combine_speaker=combine_speaker,
        )

    for include_backchannels in [True, False]:
        for include_overlaps in [True, False]:
            for include_yield_token in [True, False]:
                spoken_dm = SpokenDM(
                    datasets=dataset,
                    process="aligned",
                    include_backchannels=include_backchannels,
                    include_overlaps=include_overlaps,
                    include_yield_token=include_yield_token,
                )
                spoken_dm.prepare_data()
