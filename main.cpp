#include <iostream>
#include <string>
#include <vector>

#include "raylib.h"
#include "raymath.h"
#include "rcamera.h"

#include "linear-regression.h"

#define WIDTH  (((float)1920 * 0.75f))
#define HEIGHT (((float)1080 * 0.75f))

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
  TwoVariableLinearRegression lr;
  float p1, p2, p3, p4;

  // Initialization
  const int screenWidth  = WIDTH;
  const int screenHeight = HEIGHT;

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

    if (IsKeyPressed(KEY_R)) {
      camera.target = (Vector3){0.0f, 0.0f, 0.0f}; // Camera looking at point
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
        lr.push(collision.point.x, collision.point.z, current_y);

        auto [b0, b1, b2] = lr.solve();

        p1 = b0 + GRID_SIZE * b1 + GRID_SIZE * b2;
        p2 = b0 + GRID_SIZE * b1 + -GRID_SIZE * b2;
        p3 = b0 + -GRID_SIZE * b1 + -GRID_SIZE * b2;
        p4 = b0 + -GRID_SIZE * b1 + GRID_SIZE * b2;

        UnloadModel(plane_model);
        plane_mesh  = GenMeshCustomPlane(p1, p2, p3, p4);
        plane_model = LoadModelFromMesh(plane_mesh);
      }
    }
    for (int i = 0; i < lr.size(); i++) {
      Vector3 v = {(float)lr.get()[i][0], (float)lr.get()[i][2], (float)lr.get()[i][1]};
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