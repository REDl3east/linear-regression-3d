#include <iostream>
#include <string>
#include <vector>

// #include "xtensor-blas/xlinalg.hpp"
// #include "xtensor/xadapt.hpp"
// #include "xtensor/xio.hpp"
// #include "xtensor/xmath.hpp"
// #include "xtensor/xtensor.hpp"
// #define PI xt::numeric_constants<double>::PI

#include "raylib.h"
#include "raymath.h"
#include "rcamera.h"

#define WIDTH  (((float)1920 / 1.5))
#define HEIGHT (((float)1080 / 1.5))

#define GRID_SIZE                     100
#define CAMERA_ROTATION_SPEED         0.03f
#define CAMERA_ORBITAL_SPEED          0.5f // Radians per second
#define CAMERA_MOVE_SPEED             0.25f
#define CAMERA_PAN_SPEED              0.2f
#define CAMERA_MOUSE_MOVE_SENSITIVITY 0.003f // TODO: it should be independant of framerate

void CustomUpdateCamera(Camera* camera) {
  Vector2 mousePositionDelta = GetMouseDelta();
  int mode                   = CAMERA_FREE;

  bool moveInWorldPlane   = false;
  bool rotateAroundTarget = false;
  bool lockView           = false;
  bool rotateUp           = false;

  if (mode == CAMERA_ORBITAL) {
    // Orbital can just orbit
    Matrix rotation  = MatrixRotate(GetCameraUp(camera), CAMERA_ORBITAL_SPEED * GetFrameTime());
    Vector3 view     = Vector3Subtract(camera->position, camera->target);
    view             = Vector3Transform(view, rotation);
    camera->position = Vector3Add(camera->target, view);
  } else {
    // Camera rotation
    if (IsKeyDown(KEY_DOWN)) CameraPitch(camera, -CAMERA_ROTATION_SPEED, lockView, rotateAroundTarget, rotateUp);
    if (IsKeyDown(KEY_UP)) CameraPitch(camera, CAMERA_ROTATION_SPEED, lockView, rotateAroundTarget, rotateUp);
    if (IsKeyDown(KEY_RIGHT)) CameraYaw(camera, -CAMERA_ROTATION_SPEED, rotateAroundTarget);
    if (IsKeyDown(KEY_LEFT)) CameraYaw(camera, CAMERA_ROTATION_SPEED, rotateAroundTarget);

    // Camera movement
    if (!IsGamepadAvailable(0)) {
      // Camera pan (for CAMERA_FREE)
      if ((mode == CAMERA_FREE) && (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))) {
        const Vector2 mouseDelta = GetMouseDelta();
        if (mouseDelta.x > 0.0f) CameraMoveRight(camera, CAMERA_PAN_SPEED, moveInWorldPlane);
        if (mouseDelta.x < 0.0f) CameraMoveRight(camera, -CAMERA_PAN_SPEED, moveInWorldPlane);
        if (mouseDelta.y > 0.0f) CameraMoveUp(camera, -CAMERA_PAN_SPEED);
        if (mouseDelta.y < 0.0f) CameraMoveUp(camera, CAMERA_PAN_SPEED);
      } else {
        // Mouse support
        CameraYaw(camera, -mousePositionDelta.x * CAMERA_MOUSE_MOVE_SENSITIVITY, rotateAroundTarget);
        CameraPitch(camera, -mousePositionDelta.y * CAMERA_MOUSE_MOVE_SENSITIVITY, lockView, rotateAroundTarget, rotateUp);
      }

      // Keyboard support
      if (IsKeyDown(KEY_W)) CameraMoveForward(camera, CAMERA_MOVE_SPEED, moveInWorldPlane);
      if (IsKeyDown(KEY_A)) CameraMoveRight(camera, -CAMERA_MOVE_SPEED, moveInWorldPlane);
      if (IsKeyDown(KEY_S)) CameraMoveForward(camera, -CAMERA_MOVE_SPEED, moveInWorldPlane);
      if (IsKeyDown(KEY_D)) CameraMoveRight(camera, CAMERA_MOVE_SPEED, moveInWorldPlane);
    } else {
      // Gamepad controller support
      CameraYaw(camera, -(GetGamepadAxisMovement(0, GAMEPAD_AXIS_RIGHT_X) * 2) * CAMERA_MOUSE_MOVE_SENSITIVITY, rotateAroundTarget);
      CameraPitch(camera, -(GetGamepadAxisMovement(0, GAMEPAD_AXIS_RIGHT_Y) * 2) * CAMERA_MOUSE_MOVE_SENSITIVITY, lockView, rotateAroundTarget, rotateUp);

      if (GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_Y) <= -0.25f) CameraMoveForward(camera, CAMERA_MOVE_SPEED, moveInWorldPlane);
      if (GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X) <= -0.25f) CameraMoveRight(camera, -CAMERA_MOVE_SPEED, moveInWorldPlane);
      if (GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_Y) >= 0.25f) CameraMoveForward(camera, -CAMERA_MOVE_SPEED, moveInWorldPlane);
      if (GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X) >= 0.25f) CameraMoveRight(camera, CAMERA_MOVE_SPEED, moveInWorldPlane);
    }

    if (mode == CAMERA_FREE) {
      if (IsKeyDown(KEY_SPACE)) CameraMoveUp(camera, CAMERA_MOVE_SPEED);
      if (IsKeyDown(KEY_LEFT_CONTROL)) CameraMoveUp(camera, -CAMERA_MOVE_SPEED);
    }
  }
}

int main(int, char**) {
  std::vector<float> x_data;
  std::vector<float> y_data;
  std::vector<float> z_data;

  // Initialization
  const int screenWidth  = 800;
  const int screenHeight = 450;

  InitWindow(screenWidth, screenHeight, "Linear Regression 3D");

  // Define the camera to look into our 3d world
  Camera3D camera   = {0};
  camera.position   = (Vector3){25.0f, 25.0f, 25.0f}; // Camera position
  camera.target     = (Vector3){0.0f, 0.0f, 0.0f};    // Camera looking at point
  camera.up         = (Vector3){0.0f, 1.0f, 0.0f};    // Camera up vector (rotation towards target)
  camera.fovy       = 45.0f;                          // Camera field-of-view Y
  camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type

  Ray ray                = {0}; // Picking line ray
  RayCollision collision = {0}; // Ray collision hit info

  Mesh mesh   = GenMeshPlane(GRID_SIZE, GRID_SIZE, 4, 3);
  Model model = LoadModelFromMesh(mesh);

  float current_y = 0.0f;
  DisableCursor();

  SetTargetFPS(60);

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_Q)) break;
    CustomUpdateCamera(&camera);

    if (GetMouseWheelMoveV().y > 0) {
      current_y += 0.25;
    }
    if (GetMouseWheelMoveV().y < 0) {
      current_y -= 0.25;
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);

    BeginMode3D(camera);
    collision = GetRayCollisionMesh(GetMouseRay({GetScreenWidth() * 0.5f, GetScreenHeight() * 0.5f}, camera), mesh, model.transform);
    if (collision.hit) {
      Vector3 v1 = {collision.point.x, current_y, collision.point.z};
      Vector3 v2 = collision.point;
      DrawSphere(v1, 0.5, RED);
      DrawSphere(v2, 0.25, ColorAlpha(GREEN, 0.5));
      DrawLine3D(v1, v2, YELLOW);
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
      if (collision.hit) {
        x_data.push_back(collision.point.x);
        y_data.push_back(current_y);
        z_data.push_back(collision.point.z);
        TraceLog(LOG_INFO, "%f %f %f", collision.point.x, current_y, collision.point.z);
      }
    }
    for (int i = 0; i < x_data.size(); i++) {
      Vector3 v = {x_data[i], y_data[i], z_data[i]};
      DrawSphere(v, 0.5, RED);
    }
    DrawGrid(GRID_SIZE, 1.0f);
    EndMode3D();

    EndDrawing();
  }

  UnloadModel(model);
  CloseWindow();

  return 0;
}