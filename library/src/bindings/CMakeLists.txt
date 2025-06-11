#####################
## Python Bindings ##
#####################

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; import pathlib; print(pathlib.Path(nanobind.cmake_dir()).parent)"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG)


if(nanobind_FOUND)
    # Add source files
    nanobind_add_module(_subjective_logic_lib_python_api
            # single opinion type bindings
            bindings.cpp

            types/cuda_compatible_array.cpp
            types/dirichlet_distribution.cpp

            opinions/opinion_no_base.cpp
            opinions/opinion.cpp
            opinions/trusted_opinion.cpp

            # multi source bindings
            multi_source/fusion_operators.cpp
            multi_source/conflict_operators.cpp
            multi_source/trust_revision_operators.cpp
            multi_source/trusted_fusion_operators.cpp
    )
    foreach (nanobind_name nanobind nanobind-static nanobind-abi3)
        if (TARGET ${nanobind_name})
            target_compile_options(${nanobind_name} PRIVATE -Wno-shadow)
        endif()
    endforeach ()
    target_compile_options(_subjective_logic_lib_python_api PRIVATE -Wno-shadow)
    target_compile_options(_subjective_logic_lib_python_api PRIVATE -Wno-pedantic)

    # Link libraries
    target_link_libraries(_subjective_logic_lib_python_api
        PRIVATE
            ${PROJECT_NAME}::${LIBRARY_NAME}
    )

    # Configure
    target_compile_features(_subjective_logic_lib_python_api PRIVATE cxx_std_17)


    if (aduulm_cmake_tools_FOUND)
      # Install
      install(TARGETS _subjective_logic_lib_python_api
              LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/subjective_logic COMPONENT Runtime
              )

      ## Python module
      install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/subjective_logic/
              DESTINATION ${PYTHON_INSTALL_DIR}/subjective_logic)

      nanobind_add_stub(
              _subjective_logic_lib_python_api_stub
              MODULE _subjective_logic_lib_python_api
              OUTPUT _subjective_logic_lib_python_api.pyi
              PYTHON_PATH ${PYTHON_INSTALL_DIR}/subjective_logic
              DEPENDS _subjective_logic_lib_python_api
      )
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_subjective_logic_lib_python_api.pyi
              DESTINATION ${PYTHON_INSTALL_DIR}/subjective_logic COMPONENT Runtime
      )

    elseif(SKBUILD)
      # Install
      install(TARGETS _subjective_logic_lib_python_api
              LIBRARY DESTINATION subjective_logic COMPONENT Runtime
              )

      ## Python module
      install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/subjective_logic/
              DESTINATION subjective_logic)
    endif()

else()
    message(WARNING "Nanobind not found, skip building python bindings for subjective_logic")
Endif()
