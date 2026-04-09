enable_language(CUDA)

add_executable(multi_operator_test_cu
        multi_source_conflict.cu
        multi_source_fusion.cu
        multi_source_trust_revision.cu
        multi_source_trusted_fusion.cu
        test.cu
)


foreach(target multi_operator_test_cu)
    target_link_libraries(${target}
            PUBLIC
            -lstdc++
            subjective_logic_lib::subjective_logic_lib
    )

    target_compile_features(${target} PUBLIC cxx_std_20)
    target_compile_options(${target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
#        target_compile_options(${target} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_features(${target} PUBLIC cuda_std_20)

    set(CUDA_TEST_TARGETS "${CUDA_TEST_TARGETS};${target}" PARENT_SCOPE)
endforeach()


