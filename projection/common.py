import sys
from os import makedirs, path

from stable_baselines3.common.logger import Logger, HumanOutputFormat, WandbWriter, CSVOutputFormat


def make_logger(project: str, run_name: str) -> Logger:
    dir_path = path.abspath(path.join(__file__, '..', '..', 'output', project))
    makedirs(dir_path, exist_ok=True)

    csv_path = path.abspath(path.join(dir_path, f'{run_name}.csv'))

    return Logger(
        folder='../.logs',
        output_formats=[
            HumanOutputFormat(sys.stdout),
            WandbWriter(),
            CSVOutputFormat(csv_path)
        ],
    )