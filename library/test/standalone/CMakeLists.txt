
if (${COVERAGE})
#    by linking the subjective_logic_lib to the operator tests, they automatically get ftest_coverage flagged during coverage builds...
#    as a result, gcovr tries to parse those files and becomes super unhappy...
#    thus, they are simply omitted...
    MESSAGE(WARNING "Skipping Operator test build in Coverage mode, due to errors with cuda compiled files and gcovc")
else ()
    set(SL_OPERATOR_TEST_FILES
            operator_test.cpp
    )
    add_executable(operator_test
            ${SL_OPERATOR_TEST_FILES}
    )

    foreach(target operator_test)
        target_link_libraries(${target}
                PUBLIC
                subjective_logic_lib::subjective_logic_lib
        )

        target_compile_features(${target} PUBLIC cxx_std_20)
    endforeach()

    set(CUDA_TEST_TARGETS "")
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}\nbuilding CUDA tests for subjective_logic_lib")
        enable_language(CUDA)
        set(SL_OPERATOR_TEST_CU_FILES
                operator_test.cu
        )
        add_executable(operator_test_cu
                ${SL_OPERATOR_TEST_CU_FILES}
        )

        foreach(target operator_test_cu)
            target_link_libraries(${target}
                    PUBLIC
                    -lstdc++
                    subjective_logic_lib::subjective_logic_lib
            )

            target_compile_features(${target} PUBLIC cxx_std_20)
            target_compile_options(${target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
            target_compile_features(${target} PUBLIC cuda_std_20)
        endforeach()
        set(CUDA_TEST_TARGETS "operator_test_cu")

        add_subdirectory(multi_source)
    else()
        message(WARNING "CUDA compiler not found, skipping CUDA tests")
    endif()

    set(UNITTEST_PLAYGROUND_EXECUTABLES "${UNITTEST_PLAYGROUND_EXECUTABLES};operator_test;${CUDA_TEST_TARGETS}" PARENT_SCOPE)
endif ()

