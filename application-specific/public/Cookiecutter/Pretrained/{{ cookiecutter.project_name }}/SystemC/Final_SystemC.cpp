#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/data/example.h>
#include <torch/csrc/jit/serialization/import.h>
#include "systemc.h"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include <opencv2/opencv.hpp>
#include <random>
#include <fstream>
#include <ctime>
#include "Initiator.h"
#include "Target.h"

using namespace std;
using namespace sc_core;
using namespace tlm;

void setMKLThreads()
{
    int numThreads = 4; // Set the desired number of threads
    torch::set_num_threads(numThreads);
}

int sc_main(int argc, char *argv[])
{
  //  cout << "\nOmar\n";

    setMKLThreads();
    // Create instances of the initiator and target modules and give a name to the instance
    Initiator initiator("initiator");
    Target target("target");
    //  Connect the sockets
    initiator.init_socket.bind(target.target_socket);
    // Start simulation
    sc_start();
    // system("pause");
    return 0;
}
