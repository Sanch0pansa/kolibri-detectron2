import yaml, os
from PIL import Image
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import cv2


class Dataset:
    def __init__(self, path_to_yaml, segmentation=False):
        self.path_to_yaml = path_to_yaml
        self.path_to_dataset = os.path.dirname(path_to_yaml)
        self.segmentation = segmentation

        self.config = {
            "train": "./images/train/",
            "val": "./images/valid/",
            "nc": 2,
            "names": ['defect', 'output']
        }

        with open(self.path_to_yaml) as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.class_number = self.config['nc']
        self.class_names = self.config['names']

        # Initializing path to images
        self.train_images_path = os.path.join(
            self.path_to_dataset,
            self.config['train']
        )

        self.valid_images_path = os.path.join(
            self.path_to_dataset,
            self.config['val']
        )

        # Initializing path to labels
        self.train_labels_path = os.path.join(
            self.path_to_dataset,
            self.config['train'].replace("images", "labels")
        )

        self.valid_labels_path = os.path.join(
            self.path_to_dataset,
            self.config['val'].replace("images", "labels")
        )

        self.version = "1"

        self.train_dataset = self.load_dataset("train")
        self.valid_dataset = self.load_dataset("val")

    def get_annotations(self, label_path, width, height):
        annotations = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                category_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                xbr = x + w / 2
                ybr = y + h / 2
                xtl = xbr - w
                ytl = ybr - h

                xtl = int(xtl * width)
                ytl = int(ytl * height)
                xbr = int(xbr * width)
                ybr = int(ybr * height)
                record = {
                    'bbox': [
                        xtl,
                        ytl,
                        xbr,
                        ybr],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': category_id
                }
                if self.segmentation:
                    record['segmentation'] = [[xtl, ytl, xbr, ytl, xbr, ybr, xtl, ybr]]
                annotations.append(record)
        return annotations

    def load_dataset(self, phase):
        dataset = []
        labels_path = self.train_labels_path
        images_path = self.train_images_path

        if phase == 'val':
            images_path = self.valid_images_path
            labels_path = self.valid_labels_path

        for txt_file in os.listdir(labels_path):
            if txt_file.endswith('.txt'):
                image_id = txt_file.split('.')[0]
                image_file = os.path.join(images_path, f"{image_id}.jpg")
                width, height = Image.open(image_file).size
                image_info = {
                    'file_name': image_file,
                    'image_id': int(image_id),
                    'height': height,
                    'width': width,
                    'annotations': self.get_annotations(os.path.join(labels_path, txt_file), width, height)
                }
                dataset.append(image_info)

        return dataset

    @property
    def train_length(self):
        return len(os.listdir(self.train_images_path))

    @property
    def val_length(self):
        return len(os.listdir(self.train_images_path))

    def register(self, version="1"):
        self.version = version
        for phase in ['train', 'val']:
            DatasetCatalog.register(f"kolibri_{phase}-{version}", lambda p=phase: self.load_dataset(p))
            MetadataCatalog.get(f"kolibri_{phase}-{version}").set(thing_classes=self.class_names)

    @property
    def train_dataset_name(self):
        return f"kolibri_train-{self.version}"

    @property
    def train_valid_name(self):
        return f"kolibri_val-{self.version}"

    def get_metadata(self, phase="train"):
        return MetadataCatalog.get(f"kolibri_{phase}-{self.version}")

    def visualize_sample(self, phase="train", number=0, items_numbers=None):
        dataset = self.train_dataset if phase == "train" else self.valid_dataset

        sample = random.sample(dataset, number)
        if items_numbers is not None:
            sample = [dataset[i] for i in items_numbers]

        for d in sample:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=self.get_metadata(), scale=1)
            out = visualizer.draw_dataset_dict(d)

            img = Image.fromarray(cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB))
            img.show(f"train_img{d['image_id']}")


if __name__ == '__main__':
    d = Dataset("../data/updated_dataset/dataset.yaml")
    d.register()
    d.visualize_sample("train", items_numbers=[0,])