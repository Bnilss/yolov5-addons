import sys
import time
from pathlib import Path

# import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(str(FILE.parents[0]))  # add yolov5/ to path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, check_requirements, check_imshow, colorstr, is_ascii, 
            non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer, save_one_box)
from utils.plots import Annotator, colors
# from utils.torch_utils import select_device, load_classifier, time_sync

opt = dict(
    weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
)

annot_params = dict(
    lw=5,
    hide_lbl=False,
    hide_conf=True
)

class Yolov5:
    def __init__(self, 
            img_s=opt["imgsz"], weights=opt["weights"], device=opt["device"], 
            conf=opt['conf_thres'], iou=opt["iou_thres"], max_det=opt["max_det"]) -> None:
        self.annot_kws = annot_params
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.device = torch.device(device)
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.half = opt["half"] and (self.device.type != "cpu")
        self.dataset = None
        self.crops = {}
        if self.half:
            self.model.half()
        else:
            self.model.float()
        self.names = self.model.names
        img_s = (img_s,) * 2 if isinstance(img_s, int) else img_s
        self.imgsz = check_img_size(img_s, s=self.stride)
        self.ascii = is_ascii(self.names)


    def __init_inputs(self, source, is_array):
        if is_array: self.webcam = False
        else: self.webcam = not is_array and (source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://')))
        if self.webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=True)
            bs = len(self.dataset)  # batch_size
        else:
            self.webcam = False
            self.dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=True, 
                                    is_array=is_array)
            bs = 1  # batch_size
        self.vid_w, self.vid_p = [None] * bs, [None] * bs

    def _process_pred(self, pred, img, im0s, draw_img=True, return_res=False, crops=None):
        res = {}
        for i, det in enumerate(pred): # detections per image
            if self.webcam:  # batch_size >= 1
                s, im0, frame = im0s[i].copy(), self.dataset.count
            else:
                s, im0, frame = "", im0s.copy(), 0

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if crops is not None else im0
            annot = Annotator(im0, line_width=self.annot_kws["lw"], pil=not self.ascii)
            if not len(det): return
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"#{n} {self.names[int(c)]}{'s' * (n > 1)}| "  # add to string
            coords = {}
            print(f"({i+1})", "found:", s)
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                cl = self.names[c]
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                coords[cl] = coords.get(cl, [])
                
                coords[cl].append((xywh, conf))
                if draw_img:
                    label = None if self.annot_kws["hide_lbl"] else (cl if self.annot_kws["hide_conf"] else f'{cl} {conf:.2f}')
                    annot.box_label(xyxy, label, color=colors(c, True))
                if crops is not None:
                    crops[cl] = crops.get(cl, [])
                    crops[cl].append( save_one_box(xyxy, imc, "", save=False, BGR=True) )
            if return_res:
                res[i] = coords
        return annot.result(), res

    def one_batch(self, source, is_array, ind=0):
        if self.dataset is None:
            self.__init_inputs(source, is_array)
        img, im0 = [i for i in self.dataset][ind][1:3]
        return img[None] / 255.0, im0
                
                
    def detect(self, source, is_array=False, process_preds=True, as_gen=0,
            crop_det=False, only_clss=None):
        # prepare inputs
        self.__init_inputs(source, is_array)
        results = []
        t0 = time.time()
        i = 0
        self.crops.clear()
        if only_clss and set(only_clss).intersection(self.names):
            only_clss = [self.names.index(c) for c in only_clss]
        for path, img, img0s, cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img = img / 255.0
            if len(img.shape) == 3:
                img = img[None]
            #get predictions
            pred = self.model(img, augment=False, visualize=False)[0]
            #nms
            pred = non_max_suppression(pred, self.conf, self.iou, only_clss, max_det=self.max_det)
            # pred: (xm,ym,xmx,ymx,conf,cls)
            if process_preds:
                # save crops
                crops = None
                if crop_det and not self.webcam:
                    self.crops[i] = crops = {}
                res_img, _ = self._process_pred(pred, img, img0s, crops=crops)
                results.append(res_img)
                if i and as_gen and i % as_gen == 0:
                    yield tuple(self.crops.values()) if crop_det else results
                    self.crops.clear(); results.clear()
            else:
                results.append(pred)
            i += 1
        if as_gen:
            yield tuple(self.crops.values()) if crop_det else results
        print(f'Done. ({time.time() - t0:.3f}s)')
        return results
            

    def get_crops(self, ind=None):
        if self.crops:
            print(f"[INFO] There exists crops from {len(self.crops)} images")
            if ind is None:
                return self.crops.copy()
            return self.crops[ind].copy()
        print("[WARNING] There is no crops found!")
        return
