cmake_minimum_required(VERSION 3.25.2)
project(subjectiv_logic_lib_examples LANGUAGES CXX)

OPTION(USE_CUDA "set if cuda should be used, if available" ON)
OPTION(RUN_DST "set if the dst evaluation should be included" OFF)

include(CheckLanguage)
check_language(CUDA)
set(BUILD_CUDA ${USE_CUDA})
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else()
    message(WARNING "CUDA not found, skipping cuda dependent targets/functions")
    set(BUILD_CUDA OFF)
endif()

#find_package(aduulm_cmake_tools REQUIRED)
find_package(subjective_logic_lib REQUIRED)

include(FetchContent)
FetchContent_Declare(
        slAtFzi
        GIT_REPOSITORY https://github.com/fzi-forschungszentrum-informatik/SUBJ.git
        GIT_TAG 516117b
)
FetchContent_MakeAvailable(slAtFzi)

FetchContent_Declare(
        dstAtHeudiasyc
        GIT_REPOSITORY https://github.com/heudiasyc/efficient-DST.git
        GIT_TAG 90e7df8
)

FetchContent_MakeAvailable(dstAtHeudiasyc)
execute_process(
        COMMAND git apply --ignore-whitespace  ${CMAKE_CURRENT_LIST_DIR}/dst_update.patch
        COMMAND git status
        WORKING_DIRECTORY "${dstatheudiasyc_SOURCE_DIR}"
        COMMAND_ECHO STDOUT
        COMMAND_ERROR_IS_FATAL LAST
)

add_library(dstAtHeudiasyc INTERFACE)
target_include_directories(dstAtHeudiasyc INTERFACE ${dstatheudiasyc_SOURCE_DIR}/include)


set(target_files
    cpu_assessment.cpp
    fzi_assessment.cpp
    heudiasyc_assessment.cpp
    grid_self_assessment.cpp
)
if (BUILD_CUDA)
    set(target_files
        ${target_files}
        gpu_assessment.cu
    )
endif ()

set(TEST_NAME grid_self_assessment)
add_executable(${TEST_NAME} ${target_files})


foreach(target ${TEST_NAME})
    target_link_libraries(${target}
            PUBLIC
            -lstdc++
            subjective_logic_lib::subjective_logic_lib
            subj
            dstAtHeudiasyc
    )
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_compile_features(${target} PUBLIC cuda_std_20)
    target_compile_options(${target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    if (NOT BUILD_CUDA)
        target_compile_definitions(${target} PUBLIC NO_GPU)
    endif()
    if (RUN_DST)
        message(WARNING "adding DST evaluation")
        target_compile_definitions(${target} PUBLIC RUN_DST)
    endif()
endforeach()

install(TARGETS grid_self_assessment
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR} COMPONENT Runtime
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR} COMPONENT Development
    RUNTIME DESTINATION ${BIN_INSTALL_DIR} COMPONENT Runtime
    PUBLIC_HEADER DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Development
    BUNDLE DESTINATION ${BIN_INSTALL_DIR} COMPONENT Runtime
)
