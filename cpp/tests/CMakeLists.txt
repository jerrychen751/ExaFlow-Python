include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0
    GIT_SHALLOW TRUE
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
# Keep GoogleTest on the same MSVC runtime as the parent project.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(exaflow_kernels_tests
    test_kernels.cpp
    test_ndarray.cpp
)
target_link_libraries(exaflow_kernels_tests PRIVATE exaflow::kernels GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(exaflow_kernels_tests)
