#include "extension.cpp"
#include "Initiator.h"
using namespace std;
using namespace sc_core;
using namespace tlm;
using namespace torch;

Initiator::Initiator(sc_core::sc_module_name name) : sc_module(name), init_socket("init_socket")
{
    SC_THREAD(sendDataThread);
}
