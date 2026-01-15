#pragma once
#ifndef TARGET_H
#define TARGET_H

#include <systemc.h>
#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/data/example.h>
#include <torch/csrc/jit/serialization/import.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace tlm;
using namespace torch;
using namespace std;
using namespace sc_core;
using namespace tlm_utils;
#include <tlm_utils/simple_target_socket.h>

struct Target : sc_module
{
    simple_target_socket<Target> target_socket;
    vector<float> flattened_array;

    SC_CTOR(Target);

    void b_transport(tlm::tlm_generic_payload &trans, sc_core::sc_time &t);
};

#endif // TARGET_H
