import argparse

DEFAULT = ['--batch-size', '4', '--pretrained', 'gpt2', '--finetune', '--cuda',
           '--include-end-bc-token', '--include-overlap-token',
           '--early-stop', '1', '--epochs', '5', '--remove-start-tokens', '--weight-tokens',
           '--overwrite',
           '--save-model-allowed',
           '--random-seed', '512',
           '--learning-rate', '0.0000625',
           '--remove-emp-metric-generation',
           '--include-speaker-embeddings', '--pretrained-learning-rate', '0.0000625']

EXPERIMENT_1 = [
    # TurnGPT: Run from gptonly/config.py
    # Single Stream
    ['--run-name', 'NO-TIME-ALIGNMENT & NO OVERLAP', '--remove-overlap', '--filter-bc-overlap-token',
     '--detail', 'Evaluate on Serialised', '--evaluate-on-serialised',
     '--no-emp-tokens', '--remove-cross-attention', '--evaluate'],
    # Serialised
    ['--run-name', 'SERIALISED',
     '--detail', 'TEST ON SERIALISED WITH ALL DATA', '--evaluate-on-serialised',
     '--filter-bc-overlap-token', '--datasets', 'switchboard',
     '--serialise-data', '--remove-overlap'],
    # Serialised Without Cross Attention
    ['--run-name', 'SERIALISED NO CROSS', '--datasets', 'switchboard',
     '--detail', 'TEST ON SERIALISED WITH ALL DATA', '--remove-cross-attention', '--evaluate-on-serialised',
     '--filter-bc-overlap-token',
     '--serialise-data', '--remove-overlap'],
]

# All evaluated on the full dataset with cross attention enabled
TIMING = [
    # Serialised
    ['--run-name', 'SERIALISED',
     '--filter-bc-overlap-token',
     '--serialise-data', '--remove-overlap', '--remove-backchannel', '--evaluate-on-full'],
    # Time Aligned Without Backchannels & Overlaps
    ['--run-name', ' TIME_ALIGNED NO OVERLAP NO BACKCHANNEL', '--overwrite',
     '--evaluate-on-full',
     '--remove-overlaps', '--remove-backchannels', '--remove-start-tokens'],
    # Time Aligned With Backchannels
    ['--run-name', ' TIME_ALIGNED NO OVERLAP',
     '--evaluate-on-full',
     '--remove-overlaps', '--remove-start-tokens'],
    # Time Aligned With Overlaps
    ['--run-name', ' TIME_ALIGNED NO BACKCHANNEL', '--overwrite',
     '--evaluate-on-full',
     '--remove-backchannels', '--remove-start-tokens'],
    # Time Aligned With Backchannels & Overlaps
    ['--run-name', 'TIME_ALIGNED',
     '--evaluate-on-full',
     '--remove-start-tokens'],
]

# Yield token enabled evaluated over the Time Aligned With Backchnnels & Overlaps dataset
EXPERIMENT_2 = [
    # Single Stream
    ['--run-name', 'YIELD NO TIME ALIGNMENT', '--no-emp-tokens',
     '--remove-overlap', '--remove-cross-attention', '--include-yield-token'],
    # Time Aligned Without Backchannels or Overlaps
    ['--run-name', 'YIELD TIME ALIGNED',
     '--include-yield-token', '--remove-backchannels', '--remove-overlaps', '--remove-start-tokens'],
    # Serialised
    ['--run-name', 'YIELD TOKEN SERIALISED', '--serialise-data', '--remove-overlaps',
     '--include-yield-token'],
    # Time Aligned With Backchannels
    ['--run-name', 'YIELD TIME ALIGNED WITH BACKCHANNELS', '--remove-overlaps',
     '--include-yield-token', '--remove-start-tokens'],
    # Time Aligned With Overlaps
    ['--run-name', 'YIELD TIME ALIGNED WITH OVERLAPS',
     '--include-yield-token', '--remove-backchannels', '--remove-start-tokens'],
    # Time Aligned With Overlaps and Backchannels
    ['--run-name', 'YIELD TIME_ALIGNED WITH BACKCHANNELS & OVERLAPS',
        '--include-yield-token', '--remove-start-tokens'],
]

# Automatically recorded from training with Experiment_2
EXPERIMENT_3 = [
    # Single Stream
    ['--run-name', 'YIELD NO TIME ALIGNMENT', '--no-emp-tokens',
     '--remove-overlap', '--remove-cross-attention', '--include-yield-token'],
    # Time Aligned Without Backchannels or Overlaps
    ['--run-name', 'YIELD TIME ALIGNED',
     '--include-yield-token', '--remove-backchannels', '--remove-overlaps', '--remove-start-tokens'],
    # Serialised
    ['--run-name', 'YIELD TOKEN SERIALISED', '--serialise-data', '--remove-overlaps',
     '--include-yield-token'],
    # Time Aligned With Backchannels
    ['--run-name', 'YIELD TIME ALIGNED WITH BACKCHANNELS', '--remove-overlaps',
     '--include-yield-token', '--remove-start-tokens'],
    # Time Aligned With Overlaps
    ['--run-name', 'YIELD TIME ALIGNED WITH OVERLAPS',
     '--include-yield-token', '--remove-backchannels', '--remove-start-tokens'],
    # Time Aligned With Overlaps and Backchannels
    ['--run-name', 'YIELD TIME_ALIGNED WITH BACKCHANNELS & OVERLAPS',
        '--include-yield-token', '--remove-start-tokens'],
]

CONFIGS = {
    'experiment1': EXPERIMENT_1,
    'experiment2': EXPERIMENT_2,
    'experiment3': EXPERIMENT_3,
}


def get_configs(parser: argparse.ArgumentParser, config_namespace: argparse.Namespace):
    result_run = config_namespace.result_run
    default = vars(config_namespace)

    configs = CONFIGS.get(result_run, None)
    another_run = CONFIGS.get('timing', {})
    configs = configs

    if configs is None:
        print(f"{result_run} not present in configs of keys {CONFIGS.keys()}")
        return []

    for config in configs:
        config.extend(DEFAULT)

        new_config = argparse.Namespace(**default)
        update = parser.parse_args(args=config, namespace=new_config)

        yield update
