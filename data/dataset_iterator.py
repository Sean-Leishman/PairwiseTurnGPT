class DatasetIterator:
    def __init__(self, primary_dataset, *secondary_datasets):
        self.primary_dataset = iter(primary_dataset)
        self.secondary_datasets = secondary_datasets

    def __iter__(self):
        for primary in self.primary_dataset:
            conv_id = primary.get("conv_id", None)
            if conv_id is None:
                speakerA = primary.get("speakerA", None)
                conv_id = (
                    speakerA.get("conv_id", None) if speakerA is not None else None
                )

            if conv_id is None:
                raise ValueError("Missing conv_id in primary dataset")
            secondaries = [x.get_by_conv_id(conv_id) for x in self.secondary_datasets]

            yield primary, secondaries

    def __next__(self):
        next_primary = next(self.primary_dataset)
        next_conv_id = next_primary.get("conv_id", None)
        if next_conv_id is None:
            speakerA = next_primary.get("speakerA", None)
            next_conv_id = (
                speakerA.get("conv_id", None) if speakerA is not None else None
            )

        if next_conv_id is None:
            raise ValueError("Missing conv_id in primary dataset")
        next_secondaries = [
            x.get_by_conv_id(next_conv_id) for x in self.secondary_datasets
        ]

        return next_primary, next_secondaries

    def next(self):
        return self.__next__()
