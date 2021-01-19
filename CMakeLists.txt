cmake_minimum_required(VERSION 3.9)

set(ProjectName "template")

option(ENABLE_OPENCV "Enable OpenCV integration" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_CUDNN "Enable CUDNN" OFF)

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

if(ENABLE_OPENCV)
  message(STATUS "[${ProjectName}] OpenCV : ON")
else()
  message(STATUS "[${ProjectName}] OpenCV : OFF")
endif()


set(Template_MAJOR_VERSION 0)
set(Template_MINOR_VERSION 2)
set(Template_PATCH_VERSION 5)
set(Template_TWEAK_VERSION 3)
set(Template_VERSION ${Template_MAJOR_VERSION}.${Template_MINOR_VERSION}.${Template_PATCH_VERSION}.${Template_TWEAK_VERSION})

project(${ProjectName} VERSION ${Template_VERSION})

enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)


set(INSTALL_BIN_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where lib will be installed")
set(INSTALL_INCLUDE_DIR  "include/${ProjectName}"           CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "share/${ProjectName}"             CACHE PATH "Path where cmake configs will be installed")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include/")

file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/include/*.h")
file(GLOB cuda_sources "${CMAKE_CURRENT_LIST_DIR}/cuda/*.cu")

add_library(${ProjectName} ${sources} ${headers} ${cuda_sources})



if(BUILD_SHARED_LIBS)
  target_compile_definitions(${ProjectName} PRIVATE LIB_EXPORTS=1)
endif()
if(BUILD_AS_CPP)
  set_target_properties(${ProjectName} PROPERTIES LINKER_LANGUAGE CXX)
endif()

set_target_properties(${ProjectName} PROPERTIES OUTPUT_NAME "template")
















include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
  set(CUDA_ARCHITECTURES "Auto" CACHE STRING "\"Auto\" detects local machine GPU compute arch at runtime, \"Common\" and \"All\" cover common and entire subsets of architectures, \"Names\" is a list of architectures to enable by name, \"Numbers\" is a list of compute capabilities (version number) to enable")
  set_property(CACHE CUDA_ARCHITECTURES PROPERTY STRINGS "Auto" "Common" "All" "Kepler Maxwell Kepler+Tegra Maxwell+Tegra Pascal" "5.0 7.5")
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
    message(STATUS "Unsupported CUDA version, please upgrade to CUDA 9+. Disabling CUDA support")
    set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA support" FORCE)
  else()
    find_package(CUDA REQUIRED)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCHITECTURES})
    message(STATUS "Building with CUDA flags: " "${CUDA_ARCH_FLAGS}")
    if (NOT "arch=compute_70,code=sm_70" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_72,code=sm_72" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_75,code=sm_75" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_80,code=sm_80" IN_LIST CUDA_ARCH_FLAGS)
      set(ENABLE_CUDNN_HALF "FALSE" CACHE BOOL "Enable CUDNN Half precision" FORCE)
      message(STATUS "Your setup does not supports half precision (it requires CC >= 7.0)")
    else()
      message(STATUS "Your setup supports half precision (it requires CC >= 7.0)")
    endif()
  endif()
  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    if(BUILD_SHARED_LIBS)
      set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
    else()
      set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
    endif()
  endif()
else()
  set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA support" FORCE)
endif()




if(ENABLE_CUDA)
  find_package(CUDNN)
  if(NOT CUDNN_FOUND)
    set(ENABLE_CUDNN "FALSE" CACHE BOOL "Enable CUDNN" FORCE)
  endif()
endif()

if(ENABLE_CUDA)
  if (MSVC)
    set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DGPU")
    if(CUDNN_FOUND)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DCUDNN")
    endif()
    if(OpenCV_FOUND)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DOPENCV")
    endif()
    string(REPLACE " " "," ADDITIONAL_CXX_FLAGS_COMMA_SEPARATED "${ADDITIONAL_CXX_FLAGS}")
    set(CUDA_HOST_COMPILER_FLAGS "-Wno-deprecated-declarations -Xcompiler=\"${ADDITIONAL_CXX_FLAGS_COMMA_SEPARATED}\"")
  else()
    set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DGPU")
    if(CUDNN_FOUND)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DCUDNN")
    endif()
    if(OpenCV_FOUND)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DOPENCV")
    endif()
    if (APPLE)
        set(CUDA_HOST_COMPILER_FLAGS "--compiler-options \" ${ADDITIONAL_CXX_FLAGS} -fPIC -Xpreprocessor -fopenmp -Ofast \"")
    else()
        set(CUDA_HOST_COMPILER_FLAGS "--compiler-options \" ${ADDITIONAL_CXX_FLAGS} -fPIC -fopenmp -Ofast \"")
    endif()
  endif()

  string (REPLACE ";" " " CUDA_ARCH_FLAGS_SPACE_SEPARATED "${CUDA_ARCH_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CUDA_ARCH_FLAGS_SPACE_SEPARATED} ${CUDA_HOST_COMPILER_FLAGS} ${CMAKE_CUDA_FLAGS}")
  message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()

if(ENABLE_CUDA)
  if(ENABLE_ZED_CAMERA)
    find_package(ZED 2 QUIET)
    if(ZED_FOUND)
      include_directories(${ZED_INCLUDE_DIRS})
      link_directories(${ZED_LIBRARY_DIR})
      message(STATUS "ZED SDK enabled")
    else()
      message(STATUS "ZED SDK not found")
      set(ENABLE_ZED_CAMERA "FALSE" CACHE BOOL "Enable ZED Camera support" FORCE)
    endif()
  endif()
else()
  message(STATUS "ZED SDK not enabled, since it requires CUDA")
  set(ENABLE_ZED_CAMERA "FALSE" CACHE BOOL "Enable ZED Camera support" FORCE)
endif()

if(BUILD_AS_CPP)
  set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
endif()










set_target_properties(${ProjectName} PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ENABLE_CUDA)
  set_target_properties(${ProjectName} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()


if(ENABLE_CUDA)
  target_include_directories(${ProjectName} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  #target_link_libraries(${ProjectName} PRIVATE curand cublas cuda)
  target_link_libraries(${ProjectName} ${CUDA_LIBRARIES})
  set_target_properties(${ProjectName} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_definitions(${ProjectName} PRIVATE -DUSE_CUDA)
endif()



#add_subdirectory(src)
add_subdirectory(test)
