#include "Render.h"

void RenderFunctions::InitialiseOpenGL(int width, int height, int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutCreateWindow("Mandlebrot Explorer");
    glClearColor(1.0, 1.0, 1.0, 1.0); // Set clear color to white
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, width, 0.0, height); // Set the coordinate system
}

void RenderFunctions::RenderArray(int* array, int width, int height, float brightness_multiplier)  
{
    int index = 0;
    for (int x = 0; x < width; ++x) 
    {
        for (int y = 0; y < height; ++y) 
        {
            int intensity = brightness_multiplier * array[index];

            std::cout << intensity;
            
            glColor3f(intensity, intensity, intensity);
            glVertex2i(x, y);
  
            ++index;
        }
    }
}

void RenderFunctions::InitialiseRender() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
}

void RenderFunctions::FinaliseRender() {
    glEnd();
    glFlush();
}
