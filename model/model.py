from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from data_process import Dataset
from multiprocessing import freeze_support
from PIL import Image
from detectron2.utils.visualizer import ColorMode
import cv2
import os


class Model:
    def __init__(self,
                 model_name: str = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
                 device: str = "cpu",
                 loader_num_workers: int = 4,
                 output_dir: str = "output",
                 load_from: str | None = None
                 ):
        self.cfg = get_cfg()

        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        self.cfg.MODEL.DEVICE = device
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))

        if load_from is None:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        else:
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, load_from)

        self.cfg.DATALOADER.NUM_WORKERS = loader_num_workers

        self.predictor = None

    def config_model(self, *args, **kwargs):
        pass

    def train(self,
              dataset_name: str,
              images_per_batch: int = 2,
              base_lr: float = 1e-3,
              gamma: float = 0.1,
              max_iterations: int = 1000,
              steps: list[int] = (250, 500, 750),
              warmup_iterations: int = 100,
              *args,
              **kwargs
              ):
        self.cfg.DATASETS.TRAIN = (dataset_name,)
        self.cfg.DATASETS.TEST = ()
        self.cfg.SOLVER.IMS_PER_BATCH = images_per_batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.GAMMA = gamma
        self.cfg.SOLVER.MAX_ITER = max_iterations
        self.cfg.SOLVER.STEPS = steps
        self.cfg.SOLVER.WARMUP_ITERS = warmup_iterations

        self.config_model(*args, **kwargs)

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def init_predictor(self):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, image):
        return self.predictor(image)

    def visualize_prediction(self, image, metadata):
        im = cv2.imread(image)
        outputs = self(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        # print(outputs["instances"])
        out = v.draw_instance_predictions(outputs["instances"])
        img = Image.fromarray(cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB))
        img.show(f"prediction")


class RCNN(Model):
    def config_model(self,
                     batch_size_per_image: int = 512,
                     num_classes: int = 2,
                     cls_agnostic_bbox_reg: bool = False,
                     train_on_prediction_boxes: bool = True,
                     ):
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = cls_agnostic_bbox_reg
        self.cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = train_on_prediction_boxes


class RetinaNet(Model):
    def config_model(self,
                     score_thresh_test: float = 0.05,
                     num_classes: int = 2
                     ):
        self.cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh_test


if __name__ == '__main__':
    freeze_support()

    # Test the process of training and prediction
    m = RCNN(
        "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
        output_dir="faster_rcnn_R_50-output",
        load_from="model_final.pth"
    )
    m.config_model()
    d = Dataset("../data/updated_dataset/dataset.yaml")
    d.register()

    # m.train(
    #     d.train_dataset_name,
    #     max_iterations=1000
    # )

    m.init_predictor()
    m.visualize_prediction(d.valid_dataset[3]['file_name'], d.get_metadata("val"))