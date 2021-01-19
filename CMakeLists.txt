cmake_minimum_required(VERSION 3.9)

set(ProjectName "template")

option(ENABLE_OPENCV "Enable OpenCV integration" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_CUDNN "Enable CUDNN" ON)

if(ENABLE_CUDA)
  message(STATUS "[${ProjectName}] Cuda : ON")
else()
  message(STATUS "[${ProjectName}] Cuda : OFF")
endif()

if(ENABLE_CUDNN)
  message(STATUS "[${ProjectName}] Cudnn : ON")
else()
  message(STATUS "[${ProjectName}] Cudnn : OFF")
endif()


set(Template_MAJOR_VERSION 0)
set(Template_MINOR_VERSION 2)
set(Template_PATCH_VERSION 5)
set(Template_TWEAK_VERSION 3)
set(Template_VERSION ${Template_MAJOR_VERSION}.${Template_MINOR_VERSION}.${Template_PATCH_VERSION}.${Template_TWEAK_VERSION})

project(template VERSION ${Template_VERSION})

enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)


set(INSTALL_BIN_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where lib will be installed")
set(INSTALL_INCLUDE_DIR  "include/${ProjectName}"           CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "share/${ProjectName}"             CACHE PATH "Path where cmake configs will be installed")

if(ENABLE_CUDA)
  file(GLOB cuda_sources "${CMAKE_CURRENT_LIST_DIR}/cuda/*.cu")
  add_subdirectory(cuda)
endif()

include_directories("${CMAKE_CURRENT_LIST_DIR}/include/")

file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/include/*.h")

add_library(template_lib ${sources} ${headers} ${cuda_sources})


#add_subdirectory(src)
add_subdirectory(test)


set_target_properties(template_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ENABLE_CUDA)
  set_target_properties(template_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
if(BUILD_SHARED_LIBS)
  target_compile_definitions(template_lib PRIVATE LIB_EXPORTS=1)
endif()
if(BUILD_AS_CPP)
  set_target_properties(template_lib PROPERTIES LINKER_LANGUAGE CXX)
endif()
set_target_properties(template_lib PROPERTIES OUTPUT_NAME "template")
