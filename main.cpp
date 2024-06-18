#include <iostream>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#define PI xt::numeric_constants<double>::PI

#include "raylib.h"
#include "raymath.h"
#include "rcamera.h"

#define WIDTH  (((float)1920 / 1.5))
#define HEIGHT (((float)1080 / 1.5))

#define GRID_SIZE                     32
#define CAMERA_ROTATION_SPEED         0.03f
#define CAMERA_ORBITAL_SPEED          0.5f // Radians per second
#define CAMERA_MOVE_SPEED             0.25f
#define CAMERA_PAN_SPEED              0.2f
#define CAMERA_MOUSE_MOVE_SENSITIVITY 0.003f // TODO: it should be independant of framerate

static Mesh GenMeshCustomPlane(float p1, float p2, float p3, float p4);

void CustomUpdateCamera(Camera* camera) {
  CameraYaw(camera, -GetMouseDelta().x * CAMERA_MOUSE_MOVE_SENSITIVITY, false);
  CameraPitch(camera, -GetMouseDelta().y * CAMERA_MOUSE_MOVE_SENSITIVITY, false, false, false);

  if (IsKeyDown(KEY_W)) CameraMoveForward(camera, CAMERA_MOVE_SPEED, false);
  if (IsKeyDown(KEY_A)) CameraMoveRight(camera, -CAMERA_MOVE_SPEED, false);
  if (IsKeyDown(KEY_S)) CameraMoveForward(camera, -CAMERA_MOVE_SPEED, false);
  if (IsKeyDown(KEY_D)) CameraMoveRight(camera, CAMERA_MOVE_SPEED, false);
  if (IsKeyDown(KEY_SPACE)) CameraMoveUp(camera, CAMERA_MOVE_SPEED);
  if (IsKeyDown(KEY_LEFT_CONTROL)) CameraMoveUp(camera, -CAMERA_MOVE_SPEED);
}

int main(int, char**) {
  std::vector<float> x_data;
  std::vector<float> y_data;
  std::vector<float> z_data;
  float p1, p2, p3, p4;

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

  Mesh grid_mesh   = GenMeshPlane(GRID_SIZE, GRID_SIZE, 4, 3);
  Model grid_model = LoadModelFromMesh(grid_mesh);

  Mesh plane_mesh   = {0};
  Model plane_model = {0};

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
    collision = GetRayCollisionMesh(GetMouseRay({GetScreenWidth() * 0.5f, GetScreenHeight() * 0.5f}, camera), grid_mesh, grid_model.transform);
    if (collision.hit) {
      Vector3 v1 = {collision.point.x, current_y, collision.point.z};
      Vector3 v2 = collision.point;
      DrawSphere(v1, 0.5, RED);
      DrawSphere(v2, 0.25, ColorAlpha(GREEN, 0.5));
      DrawLine3D(v1, v2, YELLOW);
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
      if (collision.hit) {
        TraceLog(LOG_INFO, "%f %f %f", collision.point.x, current_y, collision.point.z);

        x_data.push_back(collision.point.x);
        y_data.push_back(current_y);
        z_data.push_back(collision.point.z);

        if (x_data.size() > 1) {
          std::vector<std::size_t> shape = {1, x_data.size()};
          auto x_tensor                  = xt::transpose(xt::concatenate(xt::xtuple(xt::ones<double>(shape), xt::adapt(x_data, shape), xt::adapt(z_data, shape))));
          auto y_tensor                  = xt::transpose(xt::adapt(y_data, shape));
          auto xtranspos_x               = xt::linalg::dot(xt::transpose(x_tensor), x_tensor);
          auto xtranspose_x_inv          = xt::linalg::inv(xtranspos_x);
          auto xtranspos_y               = xt::linalg::dot(xt::transpose(x_tensor), y_tensor);
          auto a                         = xt::linalg::dot(xtranspose_x_inv, xtranspos_y);

          std::cout << a << '\n';

          float alpha = a(0, 0);
          std::cout << alpha << '\n';
          float x1 = a(1, 0);
          // float x2 = a(2, 0);

          // p1 = a(0, 0) + GRID_SIZE * a(0, 1) + GRID_SIZE * a(0, 2);
          // p2 = a(0, 0) + GRID_SIZE * a(0, 1) + -GRID_SIZE * a(0, 2);
          // p3 = a(0, 0) + -GRID_SIZE * a(0, 1) + -GRID_SIZE * a(0, 2);
          // p4 = a(0, 0) + -GRID_SIZE * a(0, 1) + GRID_SIZE * a(0, 2);

          UnloadModel(plane_model);
          plane_mesh  = GenMeshCustomPlane(p1, p2, p3, p4);
          plane_model = LoadModelFromMesh(plane_mesh);
        }
      }
    }
    for (int i = 0; i < x_data.size(); i++) {
      Vector3 v = {x_data[i], y_data[i], z_data[i]};
      DrawSphere(v, 0.5, RED);
    }
    DrawGrid(GRID_SIZE, 1.0f);

    DrawModel(plane_model, {0}, 1.0f, ColorAlpha(BLUE, 0.5));

    EndMode3D();

    EndDrawing();
  }

  UnloadModel(grid_model);
  UnloadModel(plane_model);
  CloseWindow();

  return 0;
}

static Mesh GenMeshCustomPlane(float p1, float p2, float p3, float p4) {
  Mesh mesh          = {0};
  mesh.triangleCount = 2;
  mesh.vertexCount   = mesh.triangleCount * 3;
  mesh.vertices      = (float*)MemAlloc(mesh.vertexCount * 3 * sizeof(float)); // 3 vertices, 3 coordinates each (x, y, z)
  mesh.texcoords     = (float*)MemAlloc(mesh.vertexCount * 2 * sizeof(float)); // 3 vertices, 2 coordinates each (x, y)
  mesh.normals       = (float*)MemAlloc(mesh.vertexCount * 3 * sizeof(float)); // 3 vertices, 3 coordinates each (x, y, z)

  mesh.vertices[3]  = GRID_SIZE / 2;
  mesh.vertices[4]  = p1;
  mesh.vertices[5]  = GRID_SIZE / 2;
  mesh.vertices[15] = GRID_SIZE / 2;
  mesh.vertices[16] = p1;
  mesh.vertices[17] = GRID_SIZE / 2;

  mesh.vertices[6] = GRID_SIZE / 2;
  mesh.vertices[7] = p2;
  mesh.vertices[8] = -GRID_SIZE / 2;

  mesh.vertices[0]  = -GRID_SIZE / 2;
  mesh.vertices[1]  = p3;
  mesh.vertices[2]  = -GRID_SIZE / 2;
  mesh.vertices[9]  = -GRID_SIZE / 2;
  mesh.vertices[10] = p3;
  mesh.vertices[11] = -GRID_SIZE / 2;

  mesh.vertices[12] = -GRID_SIZE / 2;
  mesh.vertices[13] = p4;
  mesh.vertices[14] = GRID_SIZE / 2;

  UploadMesh(&mesh, false);
  return mesh;
}