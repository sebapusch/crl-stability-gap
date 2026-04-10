import os
import sys

from stable_baselines3.common.logger import Logger, HumanOutputFormat, WandbWriter, CSVOutputFormat


def make_logger(project: str, run_name: str) -> Logger:
    csv_path = os.path.abspath(os.path.join(
        __file__, '..', '..', 'output', project, f'{run_name}.csv')
    )

    return Logger(
        folder='../.logs',
        output_formats=[
            HumanOutputFormat(sys.stdout),
            WandbWriter(),
            CSVOutputFormat(csv_path)
        ],
    )