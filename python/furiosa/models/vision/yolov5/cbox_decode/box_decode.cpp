#include "box_decode.hpp"

#include <cassert>
#include <iostream>

#define assertm(exp, msg) assert(((void)msg, exp))

void box_decode_feat(
    const float* const anchors, const uint32_t num_anchors, const float stride, const float conf_thres, const uint32_t max_boxes,
    const float* const feat, const uint32_t batch_size, const uint32_t ny, const uint32_t nx, const uint32_t no,
    float* const out_batch, uint32_t* out_batch_pos) {

    // assertm(false, "false");

    const uint32_t params_per_box = 6;

    const uint32_t max_out_batch_pos = params_per_box * max_boxes;

    const uint32_t nc = no - 5;

    const float* cell = feat;  // current read position

    for (uint32_t b = 0; b < batch_size; b++) {
        float* const out = out_batch + (b * max_out_batch_pos);  // move to batch
        uint32_t* const out_pos_ptr = out_batch_pos + b;  // get pointer to write position
        uint32_t out_pos = *out_pos_ptr;  // get write position

        for (uint32_t a = 0; a < num_anchors; a++) {
            const float ax = anchors[2 * a + 0] * stride;
            const float ay = anchors[2 * a + 1] * stride;

            for (uint32_t y = 0; y < ny; y++) {
                for (uint32_t x = 0; x < nx; x++) {
                    const float obj_conf = cell[4];

                    if (obj_conf > conf_thres) {
                        assertm(out_pos + params_per_box <= max_out_batch_pos, "Reached max number of boxes");

                        // compute argmax to find highest conf class
                        uint32_t cls = -1;
                        float cls_conf = 0.0f;
                        for (uint32_t c = 0; c < nc; c++) {
                            float cur_cls_conf = cell[5 + c];
                            if (cur_cls_conf > cls_conf) {
                                cls_conf = cur_cls_conf;
                                cls = c;
                            }
                        }

                        const float conf = obj_conf * cls_conf;

                        if (conf > conf_thres) {
                            // yolov5 boundingbox format(center_x,center_y,width,height)
                            float bx = cell[0];
                            float by = cell[1];
                            float bw = cell[2];
                            float bh = cell[3];

                            // (feat[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                            // (feat[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                            bx = (bx * 2.0f - 0.5f + x) * stride;
                            by = (by * 2.0f - 0.5f + y) * stride;

                            bw *= 2.0f;
                            bh *= 2.0f;
                            bw = (bw * bw) * ax;
                            bh = (bh * bh) * ay;

                            // xywh -> xyxy
                            const float bw_half = 0.5f * bw;
                            const float bh_half = 0.5f * bh;

                            const float bx1 = bx - bw_half;
                            const float bx2 = bx + bw_half;
                            const float by1 = by - bh_half;
                            const float by2 = by + bh_half;

                            // write box
                            out[out_pos + 0] = bx1;
                            out[out_pos + 1] = by1;
                            out[out_pos + 2] = bx2;
                            out[out_pos + 3] = by2;
                            out[out_pos + 4] = conf;
                            out[out_pos + 5] = cls;  // int -> float

                            // move one box forward to next write position
                            out_pos += params_per_box;
                        }
                    }

                    cell += no;
                }
            }
        }

        // update buffer end
        *out_pos_ptr = out_pos;
    }
}
