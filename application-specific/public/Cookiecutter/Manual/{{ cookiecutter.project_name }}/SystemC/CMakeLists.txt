cmake_minimum_required(VERSION 3.18)
project(Final_SystemC CXX)

#Find package fetch libraries and give an error when unable to find either of this packages
find_package(SystemCLanguage CONFIG REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS} ${SystemCLanguage_INCLUDE_DIRS} )


# Add your source files and cpp files needed
add_executable(
     Final_SystemC Final_SystemC.cpp Initiator.cpp 
     InitiatorThread.cpp Target.cpp extension.cpp
     )

set (CMAKE_CXX_STANDARD 17 ${SystemC_CXX_STANDARD} CACHE STRING
     "C++ standard to build all targets. Supported values are 98, 11, 14, and 17.")
set (CMAKE_CXX_STANDARD_REQUIRED ${SystemC_CXX_STANDARD_REQUIRED} CACHE BOOL
     "The with CMAKE_CXX_STANDARD selected C++ standard is a requirement.")
set(source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(destination_dir "${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${destination_dir})
# copies data folders needed and insert them in the destination appointed to and make it easy to reference data relative to the project
file(COPY ${source_dir}/image_samples DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${source_dir}/Pt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${source_dir}/mnist DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Final_SystemC 
PUBLIC ${OpenCV_LIBS} 
PUBLIC "${TORCH_LIBRARIES}" 
${SystemC_LIBRARIES}
PRIVATE SystemC::systemc
)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Final_SystemC
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Final_SystemC>)
endif (MSVC)
