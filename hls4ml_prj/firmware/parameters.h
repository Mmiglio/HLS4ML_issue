#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_conv.h"
#include "nnet_utils/nnet_conv_large.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/s2.h"
#include "weights/b2.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/s5.h"
#include "weights/b5.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/s8.h"
#include "weights/b8.h"
#include "weights/w9.h"
#include "weights/b9.h"

//hls-fpga-machine-learning insert layer-config
struct config2 : nnet::batchnorm_config {
    static const unsigned n_in = N_INPUT_1_1*N_INPUT_2_1;
    static const unsigned n_filt = 3;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_bias_t bias_t;
    typedef batch_normalization_scale_t scale_t;
};

struct config3 : nnet::conv1d_config {
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 2;
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_chan = N_INPUT_2_1;
    static const unsigned filt_width = 4;
    static const unsigned n_filt = N_FILT_3;
    static const unsigned stride = 1;
    static const unsigned dilation = 1;
    static const unsigned n_out = N_OUTPUTS_3;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef conv1d_bias_t bias_t;
    typedef conv1d_weight_t weight_t;
    typedef std::nullptr_t mult_config;
};

struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = N_OUTPUTS_3*N_FILT_3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

struct config5 : nnet::batchnorm_config {
    static const unsigned n_in = N_OUTPUTS_3*N_FILT_3;
    static const unsigned n_filt = 6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_1_bias_t bias_t;
    typedef batch_normalization_1_scale_t scale_t;
};

struct config6 : nnet::conv1d_config {
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned n_in = N_OUTPUTS_3;
    static const unsigned n_chan = N_FILT_3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_6;
    static const unsigned stride = 1;
    static const unsigned dilation = 1;
    static const unsigned n_out = N_OUTPUTS_6;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef conv1d_1_bias_t bias_t;
    typedef conv1d_1_weight_t weight_t;
    typedef std::nullptr_t mult_config;
};

struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = N_OUTPUTS_6*N_FILT_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

struct config8 : nnet::batchnorm_config {
    static const unsigned n_in = N_OUTPUTS_6*N_FILT_6;
    static const unsigned n_filt = 6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_2_bias_t bias_t;
    typedef batch_normalization_2_scale_t scale_t;
};

struct config9 : nnet::conv1d_config {
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned n_in = N_OUTPUTS_6;
    static const unsigned n_chan = N_FILT_6;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_9;
    static const unsigned stride = 1;
    static const unsigned dilation = 1;
    static const unsigned n_out = N_OUTPUTS_9;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef conv1d_2_bias_t bias_t;
    typedef conv1d_2_weight_t weight_t;
    typedef std::nullptr_t mult_config;
};

struct softmax_config10 : nnet::activ_config {
    static const unsigned n_in = N_OUTPUTS_9*N_FILT_9;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> exp_table_t;
    typedef ap_fixed<18,8> inv_table_t;
};


#endif
