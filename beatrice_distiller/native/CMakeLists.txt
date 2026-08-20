cmake_minimum_required(VERSION 3.25)
project(beatrice_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

set(BEATRICE_VST_DIR "${CMAKE_CURRENT_LIST_DIR}/../../beatrice-vst" CACHE PATH
    "Path to the beatrice-vst submodule")
set(BEATRICE_LIBRARY "" CACHE FILEPATH
    "Path to a platform-compatible Beatrice static library")

if(MSVC)
    # The official Windows library is built with the static MSVC runtime.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(BEATRICE_LIBRARY_NAME "beatrice.lib")
elseif(APPLE)
    set(BEATRICE_LIBRARY_NAME "libbeatrice.a")
elseif(UNIX)
    if(NOT BEATRICE_LIBRARY)
        message(FATAL_ERROR
            "On non-Apple Unix, set BEATRICE_LIBRARY to a compatible "
            "libbeatrice.a")
    endif()
else()
    message(FATAL_ERROR "Unsupported platform: expected MSVC or Unix")
endif()

if(NOT BEATRICE_LIBRARY)
    set(BEATRICE_LIBRARY
        "${BEATRICE_VST_DIR}/lib/beatricelib/${BEATRICE_LIBRARY_NAME}")
    set(BEATRICE_LIBRARY_URL
        "https://huggingface.co/fierce-cats/beatrice-2.0.0-alpha/resolve/rc.2/rc.0/${BEATRICE_LIBRARY_NAME}")
endif()

if(NOT EXISTS "${BEATRICE_LIBRARY}")
    if(NOT BEATRICE_LIBRARY_URL)
        message(FATAL_ERROR "Beatrice library not found: ${BEATRICE_LIBRARY}")
    endif()
    file(MAKE_DIRECTORY "${BEATRICE_VST_DIR}/lib/beatricelib")
    file(DOWNLOAD "${BEATRICE_LIBRARY_URL}" "${BEATRICE_LIBRARY}"
         STATUS download_status SHOW_PROGRESS)
    list(GET download_status 0 download_result)
    if(NOT download_result EQUAL 0)
        list(GET download_status 1 download_message)
        message(FATAL_ERROR "Could not download ${BEATRICE_LIBRARY_NAME}: "
                    "${download_message}")
    endif()
endif()

if(UNIX AND NOT APPLE)
    execute_process(
        COMMAND "${CMAKE_RANLIB}" "${BEATRICE_LIBRARY}"
        RESULT_VARIABLE ranlib_result
        ERROR_VARIABLE ranlib_error)
    if(NOT ranlib_result EQUAL 0)
        message(FATAL_ERROR "Could not index libbeatrice.a: ${ranlib_error}")
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