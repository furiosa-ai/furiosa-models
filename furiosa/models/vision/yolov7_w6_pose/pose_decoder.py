import numpy as np


class PythonPoseDecoder:
    def __init__(self, nc, anchors, stride, conf_thres) -> None:
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nkpt = 17
        self.no_kpt = 3 * self.nkpt  # number of outputs per anchor for keypoints
        self.anchors = anchors
        self.nl = anchors.shape[0]  # number of detection layers
        self.na = anchors.shape[1]  # number of anchors
        self.grid = [None for _ in range(self.nl)]
        self.anchor_grid = [None for _ in range(self.nl)]
        self.kpt_grid_x = [None for _ in range(self.nl)]
        self.kpt_grid_y = [None for _ in range(self.nl)]
        self.stride = stride
        self.conf_thres = conf_thres  # conf_thres

    def _xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        # where xy1=top-left, xy2=bottom-right
        y = np.empty_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _make_grid(self, nx=20, ny=20, i=0):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        grid = np.stack((xv, yv), axis=2).reshape((1, 1, ny, nx, 2)).astype(np.float32)
        anchor_grid = (
            (np.copy(self.anchors[i]) * self.stride[i]).reshape((1, self.na, 1, 1, 2))
            * np.ones((1, self.na, ny, nx, 2))
        ).astype(np.float32)

        kpt_grid_x = np.tile(grid[..., 0:1], (1, 1, 1, 1, self.nkpt))
        kpt_grid_y = np.tile(grid[..., 1:2], (1, 1, 1, 1, self.nkpt))

        return grid, anchor_grid, kpt_grid_x, kpt_grid_y

    def __call__(self, y_dets_kpts):
        z = []
        for i, y in enumerate(y_dets_kpts):
            y_det = y[..., :6]
            y_kpt = y[..., 6:]

            bs, _, ny, nx, _ = y_det.shape

            if self.grid[i] is None:
                (
                    self.grid[i],
                    self.anchor_grid[i],
                    self.kpt_grid_x[i],
                    self.kpt_grid_y[i],
                ) = self._make_grid(nx, ny, i)

            y_det = 1 / (1 + np.exp(-y_det))  # Sigmoid function
            y_det[..., 0:2] = (y_det[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y_det[..., 2:4] = (y_det[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            y_kpt[..., 0::3] = (y_kpt[..., ::3] * 2.0 - 0.5 + self.kpt_grid_x[i]) * self.stride[
                i
            ]  # xy
            y_kpt[..., 1::3] = (y_kpt[..., 1::3] * 2.0 - 0.5 + self.kpt_grid_y[i]) * self.stride[
                i
            ]  # xy
            y_kpt[..., 2::3] = 1 / (1 + np.exp(-y_kpt[..., 2::3]))  # Sigmoid function

            y = np.concatenate((y_det, y_kpt), axis=-1)
            z.append(y.reshape(bs, -1, 57))

        out = []
        z = np.concatenate(z, axis=1)

        for x in z:
            x[..., 4] *= x[..., 5]
            conf = x[..., 4]
            x = x[conf > self.conf_thres]

            box = self._xywh2xyxy(x[:, :4])
            conf = x[..., 4:5]
            kpt = x[..., 6:]

            x = np.concatenate((box, conf, np.zeros_like(conf), kpt), axis=1)
            out.append(x)

        return out
