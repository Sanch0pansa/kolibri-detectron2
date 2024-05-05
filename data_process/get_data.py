import os
from zipfile import ZipFile
import yadisk
from dotenv import load_dotenv

load_dotenv()


def load(token, path):
    client = yadisk.Client(token=token)

    with client:
        client.download(path, "updated_dataset.zip")

    with ZipFile("updated_dataset.zip", 'r') as zObject:
        zObject.extractall(path="../data")

    os.remove("updated_dataset.zip")


if __name__ == '__main__':
    load(os.environ.get('YANDEX_TOKEN'), os.environ.get("YANDEX_PATH_TO_DATASET"))