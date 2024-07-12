include(FetchContent)

FetchContent_Declare(
  raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG ae50bfa
  SOURCE_DIR ${RAYLIB_SRC_DIR}
)

FetchContent_MakeAvailable(raylib)