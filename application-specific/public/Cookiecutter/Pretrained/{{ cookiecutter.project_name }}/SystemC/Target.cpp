#include "Target.h"
#include "extension.cpp"
#include <iostream>
using namespace cv;

void showImage(Mat image) {
  namedWindow("Display window",
              cv::WindowFlags::WINDOW_NORMAL); // Create a window for display.
  imshow("Display window", image);
  waitKey(0);
}

Target::Target(sc_core::sc_module_name name)
    : sc_core::sc_module(name), target_socket("target_socket") {
  target_socket.register_b_transport(this, &Target::b_transport);
}

void Target::b_transport(tlm::tlm_generic_payload &trans, sc_core::sc_time &t) {

  // Read image from the path using imread
  auto image = imread(
      "./image_samples/img_1.jpg",
      cv::ImreadModes::
          IMREAD_GRAYSCALE); // E:/Github/Siemens-System-Level-Modelling-of-ASDLA-Graduation-Project/examples/SystemC_Ctorch/build2/image_samples/img_1.jpg
  // Use show image to show the read image
  showImage(image);

  auto module = torch::jit::load(
      "./Pt/model.pt"); //"E:/Github/Siemens-System-Level-Modelling-of-ASDLA-Graduation-Project/data/result/model5.pt");

  cout << endl << " rows" << image.rows << endl;
  auto tensor_image = torch::from_blob(image.data,
                                       {1,
                                        {{cookiecutter.misc_params.channels}},
                                        {{cookiecutter.misc_params.height}},
                                        {{cookiecutter.misc_params.width}}},
                                       torch::kByte);
  tensor_image = tensor_image.toType(at::kFloat);
  tensor_image = tensor_image.div_(255);

  torch::Device device ={%- if cookiecutter.misc_params.device.value[0:-2] =="cuda" -%}
          torch::Device(torch::kCUDA,{{cookiecutter.misc_params.device.value[-1]}});
          {%- else -%}
          torch::Device(torch::kCPU);
          {% endif %}


  tensor_image = tensor_image.to(device);
  module.to(device);

  auto result = module.forward({tensor_image}).toTensor();
  cout << endl << result << " Result was " << endl;
  auto max_result = result.max(1, true);
  auto max_index = std::get<1>(max_result).item<float>();
  // Print predication
  std::cout << "\nprinting the image prediction " << max_index << endl;

  // Reading the payload and check the transaction command
  tlm::tlm_command cmd = trans.get_command();
  my_extension *test_extension_transaction;
  trans.get_extension(test_extension_transaction);
  // cout << test_extension_transaction->id << endl;

  if (cmd != tlm::TLM_IGNORE_COMMAND) {
    trans.set_response_status(tlm::TLM_GENERIC_ERROR_RESPONSE);
    return;
  }

  trans.set_response_status(tlm::TLM_OK_RESPONSE);
}