#include "Render.cuh"

inline void InitialiseOpenGL()
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(camera->width, camera->height);
    glutCreateWindow("CUDA Ray tracing");
    glClearColor(1.0, 1.0, 1.0, 1.0); // Set clear color to white
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, camera->width, 0.0, camera->height); // Set the coordinate system
}

inline void RenderArray(int* array, int width, int height, float brightness_multiplier)  
{
    int index = 0;
    for (int x = 0; x < width; ++x) 
    {
        for (int y = 0; y < height; ++y) 
        {
            int intensity;
            intensity = brightness_multiplier * array[x + y * width];
            
            glColor3f(intensity, intensity, intensity);
            glVertex2i(x, y);
  
            ++index;
        }
    }
}
