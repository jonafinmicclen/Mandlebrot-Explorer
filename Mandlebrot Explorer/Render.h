#pragma once

// Std
#include <iostream>

// Open GL
#include <GL/glut.h>

namespace RenderFunctions
{
  void InitialiseOpenGL(int width, int height, int argc, char** argv);
  void RenderArray(int* array, int width, int height, float brightness_multiplier);
  void InitialiseRender();
  void FinaliseRender();
}
