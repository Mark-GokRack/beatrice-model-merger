cmake_minimum_required(VERSION 3.25)
project(beatrice_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

# The official Windows beatrice.lib is built with the static MSVC runtime.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(pybind11 CONFIG REQUIRED)

set(BEATRICE_VST_DIR "${CMAKE_CURRENT_LIST_DIR}/../../beatrice-vst" CACHE PATH
    "Path to the beatrice-vst submodule")
set(BEATRICE_LIBRARY "${BEATRICE_VST_DIR}/lib/beatricelib/beatrice.lib")
set(BEATRICE_LIBRARY_URL
    "https://huggingface.co/fierce-cats/beatrice-2.0.0-alpha/resolve/rc.2/rc.0/beatrice.lib")

if(NOT EXISTS "${BEATRICE_LIBRARY}")
    file(DOWNLOAD "${BEATRICE_LIBRARY_URL}" "${BEATRICE_LIBRARY}"
         STATUS download_status SHOW_PROGRESS)
    list(GET download_status 0 download_result)
    if(NOT download_result EQUAL 0)
        list(GET download_status 1 download_message)
        message(FATAL_ERROR "Could not download beatrice.lib: ${download_message}")
    endif()
endif()

pybind11_add_module(_beatrice_inference beatrice_inference.cc)
target_include_directories(_beatrice_inference PRIVATE "${BEATRICE_VST_DIR}/lib")
target_link_libraries(_beatrice_inference PRIVATE "${BEATRICE_LIBRARY}")
target_compile_features(_beatrice_inference PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(_beatrice_inference PRIVATE /utf-8 /arch:AVX2)
endif()

set_target_properties(_beatrice_inference PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/..")