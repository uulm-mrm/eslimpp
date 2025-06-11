
SET(TEST_NAME subjective_logic_test)

set(SL_VARIABLE_TEST_FILES
        # single opinion type tests
        types/cuda_compatible_array_test.cpp
        types/dirichlet_distribution_test.cpp

        opinions/binomial_opinion_no_base_test.cpp
        opinions/binomial_opinion_test.cpp
        opinions/multinomial_opinion_no_base_test.cpp
        opinions/multinomial_opinion_test.cpp
        opinions/trusted_opinion.cpp
        opinions/trinomial_spreadsheet_test.cpp

        # multi source tests
        multi_source/fusion_operators.cpp
        multi_source/conflict_operators.cpp
        multi_source/trusted_fusion_operators.cpp
        multi_source/trust_revision_operators.cpp
)
add_executable(${TEST_NAME}
    ${SL_VARIABLE_TEST_FILES}
)

set(SL_OPERATOR_TEST_FILES
    operator_test.cpp
)
add_executable(operator_test
    ${SL_OPERATOR_TEST_FILES}
)

foreach(target ${TEST_NAME} operator_test)
    target_link_libraries(${target}
      PUBLIC
        subjective_logic_lib::subjective_logic_lib
    )

    target_compile_features(${target} PUBLIC cxx_std_20)

endforeach()


if (${COVERAGE})
    message(WARNING "COVERAGE IS ON")
    target_link_libraries(${TEST_NAME}
            PRIVATE
            -fprofile-arcs -ftest-coverage
            )
    target_compile_options(${TEST_NAME} PRIVATE --coverage -g -O0)
endif()

target_link_libraries( ${TEST_NAME}
  PRIVATE
    GTest::gtest_main
)
gtest_discover_tests(${TEST_NAME})


# append target to test executables (for correct installation)
set(UNITTEST_EXECUTABLES "${UNITTEST_EXECUTABLES};${TEST_NAME}" PARENT_SCOPE)
set(UNITTEST_PLAYGROUND_EXECUTABLES "${UNITTEST_PLAYGROUND_EXECUTABLES};operator_test" PARENT_SCOPE)
