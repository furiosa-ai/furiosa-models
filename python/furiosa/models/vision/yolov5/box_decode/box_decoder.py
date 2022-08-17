import numpy as np
import pkg_resources as pkg
import torch


class BoxDecoderBase:
    def __init__(self, nc, anchors, stride, conf_thres) -> None:
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.anchors = anchors
        self.nl = anchors.shape[0]  # number of detection layers
        self.na = anchors.shape[1]  # number of anchors
        self.grid = [None for _ in range(self.nl)]
        self.anchor_grid = [None for _ in range(self.nl)]
        self.stride = stride
        self.conf_thres = conf_thres


class BoxDecoderPytorch(BoxDecoderBase):
    def __init__(self, anchors, *args, **kwargs) -> None:
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)

        super().__init__(*args, anchors=anchors, **kwargs)

    def _check_version(
        self,
        current="0.0.0",
        minimum="0.0.0",
        name="version ",
        pinned=False,
        hard=False,
        verbose=False,
    ):
        # Check version vs. required version
        current, minimum = (pkg.parse_version(x) for x in (current, minimum))
        # bool
        result = (current == minimum) if pinned else (current >= minimum)
        # string
        s = f"{name}{minimum} required by YOLOv5, but {name}{current} is currently installed"
        if hard:
            assert result, s  # assert min requirements met

        return result

    def _xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        # where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if self._check_version(
            torch.__version__, "1.10.0"
        ):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(
                [torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing="ij"
            )
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (
            (self.anchors[i].clone() * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand((1, self.na, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid

    def __call__(self, x):
        x = [torch.from_numpy(t) for t in x]

        z = []
        for i, y in enumerate(x):
            bs, _, ny, nx, _ = y.shape

            if self.grid[i] is None:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            assert self.grid[i].shape[2:4] == x[i].shape[2:4]

            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            y = y.view(bs, -1, self.no)

            z.append(y)

        out = []
        # zc = z[..., 4] > self.conf_thres  # candidates

        z = torch.cat(z, 1)

        for x in z:
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[x[..., 4] > self.conf_thres]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self._xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if n == 0:  # no boxes
                continue

            x = x.numpy()
            out.append(x)

        return out  # batch * num_boxes * 6 (xyxy, conf, cls)


class BoxDecoderC(BoxDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats):
        from box_decode import cbox_decode

        assert all(isinstance(feat, np.ndarray) for feat in feats)

        out_boxes_batched = cbox_decode(self.anchors, self.stride, self.conf_thres, feats)

        return out_boxes_batched
