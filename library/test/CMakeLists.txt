
# unittests are only built if google test is installed on the system
find_package(GTest QUIET)
if(NOT ${GTest_FOUND})
    message(STATUS "GTest not found, skipping unittests for gridfusion_lib")
else()
    enable_testing()

    set(UNITTEST_EXECUTABLES "")
    set(UNITTEST_PLAYGROUND_EXECUTABLES "")

    add_subdirectory(unittest/subjective_logic)

    # append the unittests to the global list of tests
    set(TEST_EXECUTABLES "${TEST_EXECUTABLES};${UNITTEST_EXECUTABLES}" PARENT_SCOPE)
    set(PLAYGROUND_EXECUTABLES "${PLAYGROUND_EXECUTABLES};${UNITTEST_PLAYGROUND_EXECUTABLES}" PARENT_SCOPE)
endif()
