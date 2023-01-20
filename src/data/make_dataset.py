# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import os


def transform_mnist(
    directory_in: str = "data/raw/corruptmnist",
    directory_out: str = "data/processed/corruptmnist",
):

    for filename in os.listdir(directory_in):
        f = os.path.join(directory_in, filename)
        data = np.load(f)
        data = dict(
            zip(("{}".format(item) for item in data), (data[item] for item in data))
        )
        for iter, img in enumerate(data["images"]):
            data["images"][iter] = (img - img.mean()) / img.std()

        np.savez(
            (directory_out + "/" + filename),
            images=data["images"],
            labels=data["labels"],
            allow_pickle=data["allow_pickle"],
        )

    return data


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    transform_mnist(input_filepath, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()