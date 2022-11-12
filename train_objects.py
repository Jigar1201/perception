import torch, torchvision
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
import torch
torch.cuda.empty_cache()
from detectron2.config import get_cfg
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os
from detectron2.utils.logger import setup_logger
setup_logger()

register_coco_instances("staplebox_train_dataset", {}, "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/staple_box/final_dataset/labels_coco.json", \
                                                      "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/staple_box/final_dataset/train/original_images")
register_coco_instances("staplebox_val_dataset",   {}, "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/staple_box/final_dataset/labels_coco.json", \
                                                      "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/staple_box/final_dataset/train/original_images")
# register_coco_instances("dropbox_test_dataset",  {}, "/home/mrsd-cmptr-4/Desktop/object_recognition/labels_coco.json", "/home/mrsd-cmptr-4/Desktop/object_recognition/final_dataset/train/original_images")

my_dataset_train_metadata = MetadataCatalog.get("staplebox_val_dataset")
dataset_dicts = DatasetCatalog.get("staplebox_val_dataset")

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("staplebox_val_dataset",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = [] #(1000, 1500)
cfg.SOLVER.GAMMA = 0.05


cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #your number of classes + 1
cfg.TEST.EVAL_PERIOD = 500

cfg.OUTPUT_DIR = "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/staple_box/final_dataset"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)

print("starting to train")
# trainer.train()

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
print("Output directory",cfg.OUTPUT_DIR)

predictor = DefaultPredictor(cfg)
# evaluator = COCOEvaluator("picking_train_dataset", cfg, False, output_dir="./output/picking_objects")
val_loader = build_detection_test_loader(cfg, "staplebox_val_dataset")
# inference_on_dataset(trainer.model, val_loader, evaluator)


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("staplebox_val_dataset", )
# cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("staplebox_val_dataset")

from detectron2.utils.visualizer import ColorMode
import glob

itr = 0
# output_video = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, (512, 384))

import time

for imageName in sorted(glob.glob('/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/staple_box/final_dataset/train/original_images/*jpg')):
  st = time.time()

  im = cv2.imread(imageName)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1],
                metadata=test_metadata, 
                scale=0.8)

  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  image = out.get_image()[:, :, ::-1]
  height, width, layers = image.shape
  size = (width,height)
  print("Sizeingg : ",size)
  # image = cv2.rotate(src, cv2.cv2.ROTATE_90_CLOCKWISE)
  # cv2.imwrite(f"image_{itr}.jpg",image)
  itr += 1
  print(time.time() - st)
  # output_video.write(image)
  # cv2.imshow("image_window", image)
  cv2.imwrite("preds.jpg", image)
  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #   break
    
cv2.destroyAllWindows()
# output_video.release()
