import ctypes
import os

import numpy as np

_clib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'cbox_decode.so'))


def _init():
    u32 = ctypes.c_uint32
    f32 = ctypes.c_float

    u32p = np.ctypeslib.ndpointer(dtype=u32, ndim=1, flags='C_CONTIGUOUS')
    f32p = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')

    _clib.box_decode_feat.argtypes = [
        f32p,  # anchors
        u32,  # num_anchors
        f32,  # stride
        f32,  # conf_thres
        u32,  # max_boxes
        f32p,  # feat
        u32,  # batch_size
        u32,  # ny
        u32,  # nx
        u32,  # no
        f32p,  # out_batch
        u32p,  # out_batch_pos
    ]
    _clib.box_decode_feat.restype = None


def _box_decode_feat(anchors, stride, conf_thres, max_boxes, feat, out_batch, out_batch_pos):
    bs, na, ny, nx, no = feat.shape

    _clib.box_decode_feat(
        anchors.reshape(-1),
        na,
        stride,
        conf_thres,
        max_boxes,
        feat.reshape(-1),
        bs,
        ny,
        nx,
        no,
        out_batch.reshape(-1),
        out_batch_pos,
    )


def box_decode(anchors, stride, conf_thres, feats):
    bs = feats[0].shape[0]
    max_boxes = int(1e4)

    out_batch = np.empty((bs, max_boxes, 6), dtype=np.float32)
    out_batch_pos = np.zeros(bs, dtype=np.uint32)

    for l, feat in enumerate(feats):
        _box_decode_feat(
            anchors[l], stride[l], conf_thres, max_boxes, feat, out_batch, out_batch_pos
        )

    out_boxes_batched = [boxes[: (pos // 6)] for boxes, pos in zip(out_batch, out_batch_pos)]

    return out_boxes_batched


_init()
