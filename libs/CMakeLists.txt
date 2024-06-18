include(FetchContent)

FetchContent_Declare(
  raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG ae50bfa
  SOURCE_DIR ${RAYLIB_SRC_DIR}
)

FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG 0.7.5
  SOURCE_DIR ${XTL_SRC_DIR}
)

FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG 11.0.0
  SOURCE_DIR ${XSIMD_SRC_DIR}
)

FetchContent_Declare(
  xtensor-blas
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
  GIT_TAG 0.21.0
  SOURCE_DIR ${XTENSOR_BLAS_SRC_DIR}
)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG 0.25.0
  SOURCE_DIR ${XTENSOR_SRC_DIR}
)

FetchContent_MakeAvailable(raylib xtl xtensor xsimd xtensor-blas)