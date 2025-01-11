import argparse


DEFAULT = ['--batch-size', '4', '--pretrained', 'gpt2', '--finetune', '--cuda',
           '--remove-backchannels', '--remove-overlaps', '--learning-rate', '0.0001',
           '--save-model-allowed',
           '--early-stop', '2', '--epochs', '5', '--weight-tokens', '--weight-reg-token', '0.5', '--weight-eos-token', '1',]

LEARNING_RATE = [
    ['--run-name', 'LR 0.01', '--learning-rate', '0.01'],
    ['--run-name', 'LR 0.005', '--learning-rate', '0.005'],
    ['--run-name', 'LR 0.001', '--learning-rate', '0.001'],
    ['--run-name', 'LR 0.0005', '--learning-rate', '0.0005'],
    ['--run-name', 'LR 0.0001', '--learning-rate', '0.0001'],
]

SPEAKER_TOKENS = [
    [
        '--run-name', 'SPEAKER TS & TOKEN TYPE & SPEAKER TOKEN EMBEDDING', '--individual-speaker-tokens', '--include-speaker-embeddings', '--weight-tokens',
        '--use-speaker-token-in-embedding'
    ],
    [
        '--run-name', 'SPEAKER TS & TOKEN TYPE', '--individual-speaker-tokens', '--include-speaker-embeddings', '--weight-tokens',
    ],
    [
        '--run-name', 'TS TOKENS & TOKEN TYPE', '--include-speaker-embeddings', '--weight-tokens',
    ],
    [
        '--run-name', 'TS TOKENS & TOKEN TYPE & SPEAKER TOKEN EMBEDDING', '--include-speaker-embeddings',
        '--weight-tokens', '--use-speaker-token-in-embedding',
    ],
    [
        '--run-name', 'TS TOKENS & TOKEN TYPE', '--include-speaker-embeddings', '--weight-tokens',
    ],
    [
        '--run-name', 'SPEAKER TS TOKENS', '--individual-speaker-tokens', '--weight-tokens',
    ],
    [
        '--run-name', 'TS TOKENS', '--weight-tokens',
    ],

]

WEIGHT_TOKENS = [
    [
        '--run-name', 'WEIGHT: 0.5 -> 1', '--weight-tokens',
        '--weight-reg-token', '0.5', '--weight-eos-token', '1',
    ],
    [
        '--run-name', 'WEIGHT: 0.75 -> 1', '--weight-tokens',
        '--weight-reg-token', '0.75', '--weight-eos-token', '1',
    ],
    [
        '--run-name', 'WEIGHT: 1 -> 1.25', '--weight-tokens',
        '--weight-reg-token', '1', '--weight-eos-token', '1.25',
    ],
    [
        '--run-name', 'WEIGHT: 1 -> 1.5', '--weight-tokens',
        '--weight-reg-token', '1', '--weight-eos-token', '1.5',
    ],
    [
        '--run-name', 'WEIGHT: 1 -> 1.75', '--weight-tokens',
        '--weight-reg-token', '1', '--weight-eos-token', '1.75',
    ],
    [
        '--run-name', 'WEIGHT: 1 -> 2', '--weight-tokens',
        '--weight-reg-token', '1', '--weight-eos-token', '2',
    ],
    [
        '--run-name', 'WEIGHT: 0.25 -> 1', '--weight-tokens',
        '--weight-reg-token', '0.25', '--weight-eos-token', '1',
    ],
]

# Augment with RESULT from SPEAKER_TOKENS
# And with Weighting of Tokens but possible to leave at default
REGRESSION_TIMINGS = [
    [
        '--run-name', 'PROJECTION REG LABEL: 0.25', '--normalize-time', '--projection-labels', '--weight-projection', '0.25'
    ],
    [
        '--run-name', 'PROJECTION REG LABEL: 0.5', '--normalize-time', '--projection-labels', '--weight-projection', '0.5'
    ],
    [
        '--run-name', 'PROJECTION REG LABEL: 0.75', '--normalize-time', '--weight-projection', '0.75', '--projection-labels',
    ],
    [
        '--run-name', 'PROJECTION REG LABEL: 0.05', '--normalize-time', '--weight-projection', '0.05', '--projection-labels',
    ],
    [
        '--run-name', 'PROJECTION REG LABEL: 0.10', '--normalize-time', '--weight-projection', '0.10', '--projection-labels',
    ],
    [
        '--run-name', 'PROJECTION REG LABEL: 0.15', '--normalize-time', '--weight-projection', '0.15', '--projection-labels',
    ],
    [
        '--run-name', 'PROJECTION REG LABEL: 0.20', '--normalize-time', '--weight-projection', '0.20', '--projection-labels',
    ],


    ['--run-name', 'CATEGORY TIMINGS: 0.25; BINS: 5',
     '--categorize-projection', '--weight-projection', '0.25', '--category-bins', '5',
     '--projection-labels'],
    ['--run-name', 'CATEGORY TIMINGS: 0.5; BINS: 5', '--overwrite',
     '--categorize-projection', '--weight-projection', '0.5', '--category-bins', '5',
     '--projection-labels'],
    ['--run-name', 'CATEGORY TIMINGS: 0.75; BINS: 5', '--overwrite',
     '--categorize-projection', '--weight-projection', '0.75', '--category-bins', '5',
     '--projection-labels'],
    ['--run-name', 'CATEGORY TIMINGS: 0.25; BINS: 10', '--overwrite',
     '--categorize-projection', '--weight-projection', '0.25', '--category-bins', '10',
     '--projection-labels'],
    ['--run-name', 'CATEGORY TIMINGS: 0.5; BINS: 10', '--overwrite',
     '--categorize-projection', '--weight-projection', '0.5', '--category-bins', '10',
     '--projection-labels'],

    [
        '--run-name', 'SPEAKER TS & TOKEN TYPE & SPEAKER TOKEN EMBEDDING', '--individual-speaker-tokens', '--include-speaker-embeddings', '--weight-tokens',
        '--use-speaker-token-in-embedding'
    ],
    [
        '--run-name', 'SPEAKER TS & TOKEN TYPE', '--individual-speaker-tokens', '--include-speaker-embeddings', '--weight-tokens',
    ],
    [
        '--run-name', 'TS TOKENS & TOKEN TYPE', '--include-speaker-embeddings', '--weight-tokens',
    ],
    [
        '--run-name', 'TS TOKENS & TOKEN TYPE & SPEAKER TOKEN EMBEDDING', '--include-speaker-embeddings',
        '--weight-tokens', '--use-speaker-token-in-embedding',
    ],
    [
        '--run-name', 'TS TOKENS & TOKEN TYPE', '--include-speaker-embeddings', '--weight-tokens',
    ],
    [
        '--run-name', 'SPEAKER TS TOKENS', '--individual-speaker-tokens', '--weight-tokens',
    ],
    [
        '--run-name', 'TS TOKENS', '--weight-tokens',
    ],
]

CATEGORY_TIMINGS = [
    ['--run-name', 'CATEGORY TIMINGS: 0.25; BINS: 5',
     '--categorize-projection', '--weight-projection', '0.25', '--category-bins', '5',
     '--projection-labels'],
    ['--run-name', 'CATEGORY TIMINGS: 0.5; BINS: 5', '--overwrite',
     '--categorize-projection', '--weight-projection', '0.5', '--category-bins', '5',
     '--projection-labels'],
    ['--run-name', 'CATEGORY TIMINGS: 0.75; BINS: 5', '--overwrite',
     '--categorize-projection', '--weight-projection', '0.75', '--category-bins', '5',
     '--projection-labels'],
    ['--run-name', 'CATEGORY TIMINGS: 0.25; BINS: 10', '--overwrite',
     '--categorize-projection', '--weight-projection', '0.25', '--category-bins', '10',
     '--projection-labels'],
    ['--run-name', 'CATEGORY TIMINGS: 0.5; BINS: 10', '--overwrite',
     '--categorize-projection', '--weight-projection', '0.5', '--category-bins', '10',
     '--projection-labels'],
]

RUN = [
    ['--run-name', 'DEFAULT']
]

CONFIGS = {
    'default': RUN,
    'speaker-tokens': SPEAKER_TOKENS,
    'weight-tokens': WEIGHT_TOKENS,
    'category-timings': CATEGORY_TIMINGS,
    'regression-timings': REGRESSION_TIMINGS,
    'learning-rate': LEARNING_RATE
}


def get_configs(parser: argparse.ArgumentParser, config_namespace: argparse.Namespace):
    result_run = config_namespace.result_run
    default = vars(config_namespace)

    configs = CONFIGS.get(result_run, None)
    if configs is None:
        print(f"{result_run} not present in configs of keys {CONFIGS.keys()}")
        return []

    for config in configs:
        config.extend(DEFAULT)

        new_config = argparse.Namespace(**default)
        update = parser.parse_args(args=config, namespace=new_config)

        yield update
