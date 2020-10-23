#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 10
#define N_INPUT_2_1 3
#define N_OUTPUTS_3 10
#define N_FILT_3 6
#define N_OUTPUTS_6 10
#define N_FILT_6 6
#define N_OUTPUTS_9 10
#define N_FILT_9 3

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<25,6> layer2_t;
typedef ap_fixed<16,6> batch_normalization_scale_t;
typedef ap_fixed<16,6> batch_normalization_bias_t;
typedef ap_fixed<25,6> layer3_t;
typedef ap_fixed<16,6> conv1d_weight_t;
typedef ap_fixed<16,6> conv1d_bias_t;
typedef ap_fixed<16,6> conv1d_relu_default_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<25,6> layer5_t;
typedef ap_fixed<16,6> batch_normalization_1_scale_t;
typedef ap_fixed<16,6> batch_normalization_1_bias_t;
typedef ap_fixed<25,6> layer6_t;
typedef ap_fixed<16,6> conv1d_1_weight_t;
typedef ap_fixed<16,6> conv1d_1_bias_t;
typedef ap_fixed<16,6> conv1d_1_relu_default_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<25,6> layer8_t;
typedef ap_fixed<16,6> batch_normalization_2_scale_t;
typedef ap_fixed<16,6> batch_normalization_2_bias_t;
typedef ap_fixed<25,6> layer9_t;
typedef ap_fixed<16,6> conv1d_2_weight_t;
typedef ap_fixed<16,6> conv1d_2_bias_t;
typedef ap_fixed<16,6> conv1d_2_softmax_default_t;
typedef ap_fixed<16,6> result_t;

#endif
