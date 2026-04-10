import os
import sys
from os import mkdir, path

from stable_baselines3.common.logger import Logger, HumanOutputFormat, WandbWriter, CSVOutputFormat


def make_logger(project: str, run_name: str) -> Logger:
    dir_path = path.abspath(path.join(__file__, '..', '..', 'output', project))

    if not path.exists(dir_path):
        mkdir(dir_path)

    csv_path = path.abspath(path.join(dir_path, f'{run_name}.csv'))

    return Logger(
        folder='../.logs',
        output_formats=[
            HumanOutputFormat(sys.stdout),
            WandbWriter(),
            CSVOutputFormat(csv_path)
        ],
    )