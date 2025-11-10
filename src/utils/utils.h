#include "../matrix/matrix.h"

double f(double x, double y) { return -2 * sin(x) * cos(y); }

double u(double x, double y) { return sin(x) * cos(y); }

Vector F(Vector* x, Vector* y)
{
    double h = x->data[1] - x->data[0];
    Vector R = create_vector((x->size - 2) * (y->size - 2));
    int k = 0;
    for(int i = 1; i < x->size - 1; ++i)
    {
        for(int j = 1; j < y->size - 1; ++j)
        {
            R.data[k] = f(x->data[i], y->data[j]) * h * h;
            if(i == 1)
            {
                R.data[k] -= u(x->data[0], y->data[j]);
            }
            if(i == x->size - 2)
            {
                R.data[k] -= u(x->data[x->size - 1], y->data[j]);
            }
            if(j == 1)
            {
                R.data[k] -= u(x->data[i], y->data[0]);
            }
            if(j == y->size - 2)
            {
                R.data[k] -= u(x->data[i], y->data[y->size - 1]);
            }
            ++k;
        }
    }
    return R;
}

Vector linspace(double start, double end, int n)
{
    Vector x = create_vector(n);
    for(int i = 0; i < n; i++)
    {
        x.data[i] = start + (end - start) / (n - 1) * i;
    }
    return x;
}

Vector uForXY(Vector* x, Vector* y)
{
    int k = 0;
    Vector R = create_vector((x->size - 2) * (y->size - 2));
    for(int i = 1; i < x->size - 1; ++i)
    {
        for(int j = 1; j < y->size - 1; ++j)
        {
            R.data[k] = u(x->data[i], y->data[j]);
            ++k;
        }
    }
    return R;
}
