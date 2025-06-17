cmake_minimum_required(VERSION 3.16)

project(subjective_logic_lib VERSION 0.1 LANGUAGES CXX)
enable_testing()

# Get aduulm cmake macros for coloring, package_version extraction and find_aduulm_package()
# only used within the aduulm context, e.g., setting up git hooks etc
find_package(aduulm_cmake_tools QUIET)
if (aduulm_cmake_tools_FOUND)
	message (INFO " Building inside ADUULM SBX/Context")
	get_version_string_from_package_xml(.)
	show_build_info("library")
	setup_git_hooks()
endif()

# in case nanobind can be found the library additionally builds python bindings (this requires at least python 3.8)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module QUIET)

if (Python_FOUND)
	# detect the installed nanobind package and import it into CMake
	execute_process(
			COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
			OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
	list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
	find_package(nanobind CONFIG QUIET)
else()
	set(nanobind_FOUND 0)
endif ()


# we default to Release build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(src)

# generating python bindings with nanobind adds the below checked target which causes warnings if compiled
# with -Wpedantic and -Wshadow (which are suppressed in the following because we are not interested in fixing
# those directly within nanobind)
if (TARGET nanobind-static)
	target_compile_options(nanobind-static PRIVATE -Wno-shadow -Wno-pedantic)
endif()

set(LIBRARY_TARGETS
		subjective_logic_lib
)

# UNITTESTS / STANDALONES / BENCHMARKS
# later used for installation of targets
list(APPEND TEST_EXECUTABLES "")
list(APPEND PLAYGROUND_EXECUTABLES "")

add_subdirectory(test)

# suppress some warnings caused by the nvcc compiling eigen code
foreach(target ${LIBRARY_TARGETS})
	get_target_property(type ${target} TYPE)
	if (NOT ${type} STREQUAL "INTERFACE_LIBRARY")
		target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe
				"--diag_suppress=esa_on_defaulted_function_ignored" >)
	endif ()
endforeach()

# INSTALLATION
set(PACKAGE_LIBRARY_VERSION "${package_version}")

if(aduulm_cmake_tools_FOUND)
	include(CMakePackageConfigHelpers)
	write_basic_package_version_file(
		"${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		VERSION "${PACKAGE_LIBRARY_VERSION}"
		COMPATIBILITY AnyNewerVersion
	)

	configure_package_config_file(
			"${PROJECT_SOURCE_DIR}/cmake/config.cmake.in"
			"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
			INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
			PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR BIN_INSTALL_DIR
	)
	install(TARGETS ${LIBRARY_TARGETS} ${TEST_EXECUTABLES} ${PLAYGROUND_EXECUTABLES}
			EXPORT ${PROJECT_NAME}Targets
			INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
			LIBRARY DESTINATION ${LIB_INSTALL_DIR} COMPONENT Runtime
			ARCHIVE DESTINATION ${LIB_INSTALL_DIR} COMPONENT Development
			RUNTIME DESTINATION ${BIN_INSTALL_DIR} COMPONENT Runtime
			PUBLIC_HEADER DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Development
			BUNDLE DESTINATION ${BIN_INSTALL_DIR} COMPONENT Runtime
	)

	install(EXPORT ${PROJECT_NAME}Targets
			DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
			NAMESPACE ${PROJECT_NAME}::
	)

	# install CMake config and version file
	install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
			"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
			DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME})

	# install include files
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
			DESTINATION ${INCLUDE_INSTALL_DIR})
else()
	# define std install dirs
	include (GNUInstallDirs)

	include(CMakePackageConfigHelpers)
	write_basic_package_version_file(
			"${PROJECT_NAME}ConfigVersion.cmake"
			VERSION "${PROJECT_VERSION}"
			COMPATIBILITY SameMajorVersion
	)

	configure_package_config_file(
			"${PROJECT_SOURCE_DIR}/cmake/config.cmake.in"
			"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
			INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
			NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)

	install(TARGETS ${LIBRARY_TARGETS} ${TEST_EXECUTABLES} ${PLAYGROUND_EXECUTABLES}
			EXPORT ${PROJECT_NAME}Targets
			LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Runtime
			ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
			RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
	)

	install(EXPORT ${PROJECT_NAME}Targets
			DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
			NAMESPACE ${PROJECT_NAME}::
	)

	# install CMake config and version file
	install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
				  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
			DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

	# install include files
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
			DESTINATION include)

	configure_file(
			"${PROJECT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
			"${PROJECT_BINARY_DIR}/cmake_uninstall.cmake"
			IMMEDIATE @ONLY)

	add_custom_target(uninstall
			COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

