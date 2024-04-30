import os
from zipfile import ZipFile
import yadisk
from dotenv import load_dotenv

load_dotenv()

client = yadisk.Client(token=os.environ.get('YANDEX_TOKEN'))

with client:
    client.download(os.environ.get("YANDEX_PATH_TO_DATASET"), "updated_dataset.zip")

with ZipFile("updated_dataset.zip", 'r') as zObject:
    zObject.extractall(path="../data")

os.remove("updated_dataset.zip")