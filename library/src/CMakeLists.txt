
set(LIBRARY_NAME subjective_logic_lib)
add_library(${LIBRARY_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})

target_include_directories(${LIBRARY_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

target_compile_features(${LIBRARY_NAME} INTERFACE cxx_std_20)
target_compile_definitions(${LIBRARY_NAME} INTERFACE "-D${package_name}_VERSION=\"${package_version}\"")
target_compile_options(${LIBRARY_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

if (${COVERAGE})
    message(WARNING "COVERAGE IS ON")
    target_link_libraries(${LIBRARY_NAME}
            INTERFACE
            -fprofile-arcs -ftest-coverage
            )
    target_compile_options(${LIBRARY_NAME} INTERFACE --coverage -g -O0)
endif()

# only install bindings in aduulm context, otherwise use pip to install the package directly
if (aduulm_cmake_tools_FOUND OR SKBUILD)
    add_subdirectory(bindings)
endif ()
