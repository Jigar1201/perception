from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
import torch
torch.cuda.empty_cache()
from detectron2.config import get_cfg
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os
from detectron2.utils.logger import setup_logger

# -----------------------------------------------------------------------------
setup_logger()

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

#TODO: ADD OBJECTS DATASET
cfg.DATASETS.TRAIN = ("picking_train_dataset",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = [] #(1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.OUTPUT_DIR = "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/datasets_final/souvenir_dataset_1/souvenir_dataset/final_dataset/output_dir"

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13 #your number of classes + 1
cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = CocoTrainer(cfg)
# trainer.resume_or_load(resume=False)

print("starting to train")
# trainer.train()

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

model_path = "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/datasets_final/souvenir_dataset_1/souvenir_dataset/final_dataset/output_dir/model_final.pth"

cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("picking_train_dataset")

from detectron2.utils.visualizer import ColorMode
import glob
import time
import cv2
import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib

from pdb import set_trace as bp

iter = 0
while True:
    HOST = ''
    PORT = 8485
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print('Socket created')

    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn,addr=s.accept()

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))

    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)

    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    print("Received image")
    cv2.imwrite("received_image.jpg",frame)
    im_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("received_image_bgr.jpg",im_rgb)

    outputs = predictor(im_rgb)
    v = Visualizer(im_rgb[:, :, ::-1],
                metadata=test_metadata, 
                scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = out.get_image()[:, :, ::-1]
    write_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("predictions.jpg", write_image)
    print("Outputs : ",outputs)
    preds = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy()
    scores = outputs['instances'].get_fields()['scores'].cpu().numpy()
    classes = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
    
    results = []
    results.append(preds)
    results.append(scores)
    results.append(classes)

    from pdb import set_trace as bp
    
    # IP ADDRESS
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_socket.connect(('172.26.171.22', 8485))
    connection = client_socket.makefile('wb')
    
    img_counter = 0

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    # result, frame = cv2.imencode('.jpg', frame, encode_param)
    #    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(results, 0)
    size = len(data)
    client_socket.sendall(struct.pack(">L", size) + data)
    
