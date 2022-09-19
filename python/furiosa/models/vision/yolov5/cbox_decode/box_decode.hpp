
#include <stdint.h>

extern "C" {

    void box_decode_feat(
        const float* const anchors, const uint32_t num_anchors, const float stride, const float conf_thres, const uint32_t max_boxes,
        const float* const feat, const uint32_t batch_size, const uint32_t ny, const uint32_t nx, const uint32_t no,
        float* const out_batch, uint32_t* out_batch_pos
    );

}