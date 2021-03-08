#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

struct Coords{
    float x;
    float y;
    int state;
};

float Measure(Coords a, Coords b)
{
    //distance between two points
    float new_x = (b.x - a.x)^2;
    float new_y = (b.y - a.y)^2;

    float euclidean = sqrt(new_x + new_y);

    //ratio = known distance/euclidean

    //distance = ratio * euclidean;
    

    return distance;
}
