import argparse


DEFAULT = ['--batch-size', '4', '--pretrained', 'gpt2', '--finetune', '--cuda', '--datasets', 'switchboard',
           '--remove-backchannels', '--remove-overlaps', '--learning-rate', '0.0000625',
           '--early-stop', '2', '--epochs', '5', '--weight-tokens', '--include-speaker-embeddings']

EXPERIMENT_1 = [
    ['--run-name', 'EXPERIMENT_1'],
]

RUN = [
    ['--run-name', 'DEFAULT']
]

CONFIGS = {
    'default': RUN,
    'experimen1': EXPERIMENT_1,
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
