#include "utils/utils.h"
#include <assert.h>
#include <omp.h>
#include <stdio.h>

static Matrix create_matrix_openmp(size_t rows, size_t cols)
{
    Matrix m = {rows, cols, NULL, NULL};

    m.data = (double**)malloc(rows * sizeof *m.data);
    if(!m.data)
        die("malloc data");

    size_t total;
    if(mul_overflow_size_t(rows, cols, &total))
    {
        free(m.data);
        fprintf(stderr, "Размер матрицы слишком большой (переполнение).\n");
        exit(EXIT_FAILURE);
    }

    m.buf = (double*)calloc(total, sizeof *m.buf);
    if(!m.buf)
    {
        free(m.data);
        die("calloc buf");
    }

#pragma omp parallel for
    for(size_t i = 0; i < rows; ++i)
    {
        m.data[i] = m.buf + i * cols;
    }
    return m;
}

Matrix generate_five_diag(size_t xn, size_t yn)
{

    size_t n, nn;
    if(mul_overflow_size_t(xn, yn, &n) || mul_overflow_size_t(n, n, &nn))
    {
        fprintf(stderr, "Слишком большая сетка (переполнение).\n");
        exit(EXIT_FAILURE);
    }

    Matrix A = create_matrix_openmp(n, n);
#pragma omp parallel for
    for(size_t i = 0; i < n; ++i)
    {
        // центр
        A.data[i][i] = -4.0;

        // вверх (i - xn)
        if(i >= xn)
        {
            A.data[i][i - xn] = 1.0;
        }
        // вниз (i + xn)
        if(i + xn < n)
        {
            A.data[i][i + xn] = 1.0;
        }
        // влево (i - 1) — не на левом краю строки
        if((i % xn) != 0)
        {
            A.data[i][i - 1] = 1.0;
        }
        // вправо (i + 1) — не на правом краю строки
        if((i % xn) != xn - 1)
        {
            A.data[i][i + 1] = 1.0;
        }
    }
    return A;
}

Matrix cholesky(Matrix* A, int n)
{
    Matrix L = create_matrix_openmp(n, n);
    assert(A->cols == A->rows);
    assert(A->cols == n);

    for(int j = 0; j < n; j++)
    {
        double s = 0;
        for(int k = 0; k < j; k++)
        {
            s += L.data[j][k] * L.data[j][k];
        }
        L.data[j][j] = sqrt(A->data[j][j] - s);
#pragma omp parallel for
        for(int i = j + 1; i < n; i++)
        {
            double s = 0;
            for(int k = 0; k < j; k++)
            {
                s += L.data[i][k] * L.data[j][k];
            }
            L.data[i][j] = (1.0 / L.data[j][j] * (A->data[i][j] - s));
        }
    }
    return L;
}

Vector solve_gauss_reverse(Matrix* U, Vector* b)
{
    int n = U->rows;
    Vector x = create_vector(n);

    for(int i = n - 1; i >= 0; i--)
    {
        double sum = b->data[i];

        for(int j = i + 1; j < n; j++)
        {
            sum -= U->data[i][j] * x.data[j];
        }

        if(fabs(U->data[i][i]) < 1e-12)
        {
            printf("Ошибка: нулевой диагональный элемент в строке %d!\n", i);
            free_vector(x);
            exit(EXIT_FAILURE);
        }

        x.data[i] = sum / U->data[i][i];
    }

    return x;
}

Vector solve_gauss_forward(Matrix* L, Vector* b)
{
    int n = L->rows;
    Vector x = create_vector(n);

    for(int i = 0; i < n; i++)
    {
        double sum = b->data[i];

        for(int j = 0; j < i; j++)
        {
            sum -= L->data[i][j] * x.data[j];
        }

        if(fabs(L->data[i][i]) < 1e-12)
        {
            printf("Ошибка: нулевой диагональный элемент в строке %d!\n", i);
            free_vector(x);
            exit(EXIT_FAILURE);
        }

        x.data[i] = sum / L->data[i][i];
    }

    return x;
}

Vector solve_gauss(Matrix* L, Matrix* U, Vector* b)
{
    if(L->rows != U->rows || L->rows != b->size)
    {
        printf("Ошибка: несовместимые размеры в solve_gauss!\n");
        exit(EXIT_FAILURE);
    }

    Vector y = solve_gauss_forward(L, b);

    Vector x = solve_gauss_reverse(U, &y);

    free_vector(y);

    return x;
}

Vector pcgPreconditioned(Matrix* A, Vector* b, Vector* xs, double err,
                         double* relres, int* iter, Matrix* P1, Matrix* P2)
{
    int k = 0;
    size_t n = A->cols;
    Vector x = copy_vector(xs);
    Vector r = residue(b, A, &x);

    Vector z = solve_gauss(P1, P2, &r);

    Vector p = copy_vector(&z);

    double r0norm = second_norm(&r);

    Vector current_ = create_vector(n);
    Vector newR = create_vector(n);
    Vector newZ = create_vector(n);
    Vector q = create_vector(n);
    while(second_norm(&r) / r0norm > err && k < 1000)
    {
        ++k;

        free_vector(q);
        q = matrix_vector_mult(A, &p);

        double pq = dot_product(&p, &q);

        double a = aKbyPQ(&z, &q, pq);

        free_vector(current_);
        current_ = scalar_vector_mult(&p, a);

        add_vector_self(&x, &current_);

        free_vector(current_);
        current_ = scalar_vector_mult(&q, a);

        free_vector(newR);
        newR = sub_vector(&r, &current_);

        free_vector(newZ);
        newZ = solve_gauss(P1, P2, &newR);

        double b = dot_product(&newZ, &newR) / dot_product(&z, &r);

        free_vector(z);
        z = newZ;

        free_vector(r);
        r = newR;

        free_vector(current_);
        current_ = scalar_vector_mult(&p, b);

        free_vector(p);
        p = add_vector(&r, &current_);
    }
    *iter = k;

    free_vector(z);
    free_vector(r);
    free_vector(p);
    free_vector(current_);
    free_vector(newR);
    free_vector(newZ);
    free_vector(q);
    return x;
}

void errByEpsPcgChol(double a, double b, double c, double d, double h)
{
    Vector x = linspace(a, b, (b - a) / h + 1);
    Vector y = linspace(c, d, (d - c) / h + 1);
    Vector us = uForXY(&x, &y);
    Vector B = F(&x, &y);
    int n = (x.size - 2) * (y.size - 2);
    Matrix A = generate_five_diag(x.size - 2, y.size - 2);
    printf("%zu, %zu, %i, %i, %i", A.cols, A.rows, B.size, x.size, y.size);
    scalar_mul_self(A, -1);

    scalar_vector_mult_self(&B, -1);

    printf("----\n");

    FILE* file = fopen("pcgCholErr.txt", "w");
    Matrix L = cholesky(&A, A.cols);
    Matrix Lt = transpose(L);
    Vector zeros = create_vector(n);
    for(int i = 1; i <= 10; ++i)
    {
        double eps = pow(10, -i);
        double relres = 0;
        int count = 0;

        Vector Sol =
            pcgPreconditioned(&A, &B, &zeros, eps, &relres, &count, &L, &Lt);
        printf(", iter = %i\n", count);
        double max = vectors_max_diff(&us, &Sol);
        fprintf(file, "%.15f %.15f\n", eps, max);
        printf("%.15f %.15f\n", eps, max);
        if(i == 100)
        {
            for(int j = 0; j < us.size; ++j)
            {
                printf("%.5f %.5f\n", us.data[j], Sol.data[j]);
            }
        }
        free_vector(Sol);
    }
    fclose(file);
    free_vector(x);
    free_vector(y);
    free_vector(us);
    free_vector(B);
    free_matrix(&A);
    free_matrix(&L);
    free_matrix(&Lt);
    free_vector(zeros);
}

int main()
{
    double a = 0;
    double b = 1.625;
    double c = 0;
    double d = 1.625;
    double h = 0.025;
    printf("----------------------------------------------------\n");

    errByEpsPcgChol(a, b, c, d, h);
    printf("---\n");

    return 0;
}