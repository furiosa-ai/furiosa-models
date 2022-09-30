from pathlib import Path

import cv2
import numpy as np
import torch

from furiosa.models.vision.postprocess import ObjectDetectionResult


class Yolov5Dataset:
    def __init__(self, path: Path, mode: str = "val", limit: int = None) -> None:
        path = Path(path)

        with open(path / (mode + ".txt"), "r") as f:
            img_files = [(path / l.rstrip()) for l in f.readlines()]

        if limit is not None:
            img_files = img_files[:limit]

        self.img_files = img_files
        self.label_files = [
            Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt") for p in self.img_files
        ]

        labels = []
        for label_file in self.label_files:
            with open(label_file, "r") as f:
                label = [l.rstrip().split(" ") for l in f.readlines()]
                label = np.float32(label)
                labels.append(label)

        self.labels = labels

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_files[idx]))
        boxes = np.array(self.labels[idx][:, 1:5], copy=True)
        classes = self.labels[idx][:, 0].astype(int)

        # rel xywh -> abs xyxy
        h, w = img.shape[:2]
        boxes[:, :2] = boxes[:, :2] - 0.5 * boxes[:, 2:]
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        boxes[:, 0::2] *= w
        boxes[:, 1::2] *= h
        return img, boxes, classes

    def __len__(self):
        return len(self.img_files)


def to_numpy(det_boxes: ObjectDetectionResult) -> np.ndarray:
    if len(det_boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    d = []
    for b in det_boxes:
        box = b.boundingbox
        d.append([box.left, box.top, box.right, box.bottom, b.score, b.index])
    return np.array(d, dtype=np.float32)


class MAPMetricYolov5:
    def __init__(self, num_classes) -> None:
        device = "cpu"
        self.stats = []
        self.iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.nc = num_classes

    def _box_iou(self, box1, box2):
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (
            (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2]))
            .clamp(0)
            .prod(2)
        )
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def _process_batch(self, detections, labels, iouv):
        correct = torch.zeros(
            detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device
        )
        iou = self._box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(
            (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])
        )  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            )  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        return correct

    def _ap_per_class(self, tp, conf, pred_cls, target_cls, eps=1e-16):
        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + eps)  # recall curve
                r[ci] = np.interp(
                    -px, -conf[i], recall[:, 0], left=0
                )  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = self._compute_ap(recall[:, j], precision[:, j])

        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + eps)

        i = f1.mean(0).argmax()  # max F1 index
        p, r, f1 = p[:, i], r[:, i], f1[:, i]
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives
        return tp, fp, p, r, f1, ap, unique_classes.astype('int32')

    def _compute_ap(self, recall, precision):
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def __call__(self, boxes_pred, scores_pred, classes_pred, boxes_target, classes_target):
        pred = torch.from_numpy(
            np.concatenate([boxes_pred, scores_pred[:, None], classes_pred[:, None]], 1)
        )
        labels = torch.from_numpy(np.concatenate([classes_target[:, None], boxes_target], 1))

        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(pred) == 0:
            if nl:
                self.stats.append(
                    (
                        torch.zeros(0, self.niou, dtype=torch.bool),
                        torch.Tensor(),
                        torch.Tensor(),
                        tcls,
                    )
                )
            return

        if nl:
            correct = self._process_batch(pred, labels, self.iouv)
        else:
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool)

        self.stats.append(
            (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
        )  # (correct, conf, pcls, tcls)

    def compute(self):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        assert len(stats) and stats[0].any()
        tp, fp, p, r, f1, ap, ap_class = self._ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(
            stats[3].astype(np.int64), minlength=self.nc
        )  # number of targets per class

        ap_pad = np.zeros(self.nc)
        ap_pad[nt != 0] = ap

        ap50_pad = np.zeros(self.nc)
        ap50_pad[nt != 0] = ap50

        return {
            "map": map,
            "map50": map50,
            "ap_class": ap_pad,
            "ap50_class": ap50_pad,
        }
