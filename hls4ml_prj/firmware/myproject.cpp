//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input1[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer10_out[N_OUTPUTS_9*N_FILT_9],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input1,layer10_out 
    #pragma HLS PIPELINE 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1;
    const_size_out_1 = N_OUTPUTS_9*N_FILT_9;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<batch_normalization_scale_t, 3>(s2, "s2.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 3>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv1d_weight_t, 72>(w3, "w3.txt");
        nnet::load_weights_from_txt<conv1d_bias_t, 6>(b3, "b3.txt");
        nnet::load_weights_from_txt<batch_normalization_1_scale_t, 6>(s5, "s5.txt");
        nnet::load_weights_from_txt<batch_normalization_1_bias_t, 6>(b5, "b5.txt");
        nnet::load_weights_from_txt<conv1d_1_weight_t, 108>(w6, "w6.txt");
        nnet::load_weights_from_txt<conv1d_1_bias_t, 6>(b6, "b6.txt");
        nnet::load_weights_from_txt<batch_normalization_2_scale_t, 6>(s8, "s8.txt");
        nnet::load_weights_from_txt<batch_normalization_2_bias_t, 6>(b8, "b8.txt");
        nnet::load_weights_from_txt<conv1d_2_weight_t, 54>(w9, "w9.txt");
        nnet::load_weights_from_txt<conv1d_2_bias_t, 3>(b9, "b9.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(input1, layer2_out, s2, b2);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer2_t>(layer2_out, "batch_normalization", N_INPUT_1_1*N_INPUT_2_1);
#endif

    layer3_t layer3_out[N_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::conv_1d_latency_cl<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer3_t>(layer3_out, "conv1d", N_OUTPUTS_3*N_FILT_3);
#endif

    layer4_t layer4_out[N_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<layer3_t, layer4_t, relu_config4>(layer3_out, layer4_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer4_t>(layer4_out, "conv1d_relu", N_OUTPUTS_3*N_FILT_3);
#endif

    layer5_t layer5_out[N_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::normalize<layer4_t, layer5_t, config5>(layer4_out, layer5_out, s5, b5);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer5_t>(layer5_out, "batch_normalization_1", N_OUTPUTS_3*N_FILT_3);
#endif

    layer6_t layer6_out[N_OUTPUTS_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::conv_1d_latency_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer6_t>(layer6_out, "conv1d_1", N_OUTPUTS_6*N_FILT_6);
#endif

    layer7_t layer7_out[N_OUTPUTS_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer7_t>(layer7_out, "conv1d_1_relu", N_OUTPUTS_6*N_FILT_6);
#endif

    layer8_t layer8_out[N_OUTPUTS_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::normalize<layer7_t, layer8_t, config8>(layer7_out, layer8_out, s8, b8);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer8_t>(layer8_out, "batch_normalization_2", N_OUTPUTS_6*N_FILT_6);
#endif

    layer9_t layer9_out[N_OUTPUTS_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::conv_1d_latency_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer9_t>(layer9_out, "conv1d_2", N_OUTPUTS_9*N_FILT_9);
#endif

    nnet::softmax<layer9_t, result_t, softmax_config10>(layer9_out, layer10_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer10_out, "conv1d_2_softmax", N_OUTPUTS_9*N_FILT_9);
#endif

}
