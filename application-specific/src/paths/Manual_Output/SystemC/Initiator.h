#pragma once
#ifndef INITIATOR_H
#define INITIATOR_H
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/data/example.h>
#include <torch/csrc/jit/serialization/import.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <fstream>
#include <ctime>
#include "systemc.h"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_initiator_socket.h"

using namespace tlm;
using namespace sc_core;

struct Initiator : sc_module
{
    tlm_utils::simple_initiator_socket<Initiator> init_socket;

    void sendDataThread();

    SC_CTOR(Initiator);
};

#endif // INITIATOR_H
