#include "utils/utils.h"
#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>

typedef struct
{
    int j;
    Matrix* L;
    Matrix* A;
    int start_i;
    int end_i;
} thread_data;

pthread_t* threads = NULL;
thread_data* thread_args = NULL;
int NUM_THREADS = 2;

void* compute_column(void* arg)
{
    thread_data* data = (thread_data*)arg;
    int j = data->j;
    Matrix* L = data->L;
    Matrix* A = data->A;

    for(int i = data->start_i; i < data->end_i; i++)
    {
        double s = 0.0;
        for(int k = 0; k < j; k++)
        {
            s += L->data[i][k] * L->data[j][k];
        }
        L->data[i][j] = (1.0 / L->data[j][j]) * (A->data[i][j] - s);
    }
    pthread_exit(NULL);
}

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

    // #pragma omp parallel for
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

    Matrix A = create_matrix(n, n);

    // for(size_t i = 0; i < n; ++i)
    // {
    //     printf("i: %zu ", i);
    //     work_task* task = create_task(NULL, &A, i, xn, create_matrix_helper);
    //     pool_enqueue(pool, task, 1);
    // }

    // pool_wait(pool);

    for(size_t i = 0; i < n; ++i)
    {
        // printf("i: %zu", i);
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

    pthread_t threads[NUM_THREADS];
    thread_data thread_args[NUM_THREADS];

    bool thread_created[NUM_THREADS]; // Track which threads were actually
                                      // created

    for(int j = 0; j < n; j++)
    {
        // Initialize thread tracking for this iteration
        for(int t = 0; t < NUM_THREADS; t++)
        {
            thread_created[t] = false;
        }

        // Compute diagonal element sequentially
        double s = 0.0;
        for(int k = 0; k < j; k++)
        {
            s += L.data[j][k] * L.data[j][k];
        }
        L.data[j][j] = sqrt(A->data[j][j] - s);

        // Parallelize column computation using pthreads
        int rows_remaining = n - j - 1;
        if(rows_remaining > 0)
        {
            int chunk_size = rows_remaining / NUM_THREADS;
            int extra = rows_remaining % NUM_THREADS;
            int current_start = j + 1;

            // Create threads
            for(int t = 0; t < NUM_THREADS; t++)
            {
                int thread_rows = chunk_size + (t < extra ? 1 : 0);
                if(thread_rows > 0)
                {
                    thread_args[t] =
                        (thread_data){.j = j,
                                      .L = &L,
                                      .A = A,
                                      .start_i = current_start,
                                      .end_i = current_start + thread_rows};
                    if(pthread_create(&threads[t], NULL, compute_column,
                                      &thread_args[t]) == 0)
                    {
                        thread_created[t] = true;
                    }
                    current_start += thread_rows;
                }
            }

            // Join only threads that were actually created
            for(int t = 0; t < NUM_THREADS; t++)
            {
                if(thread_created[t])
                {
                    pthread_join(threads[t], NULL);
                }
            }
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
    // printf("%zu, %zu, %i, %i, %i", A.cols, A.rows, B.size, x.size, y.size);
    scalar_mul_self(A, -1);

    scalar_vector_mult_self(&B, -1);

    // printf("----\n");

    Matrix L = cholesky(&A, A.cols);
    Matrix Lt = transpose(L);
    Vector zeros = create_vector(n);
    int i = 3;

    double eps = pow(10, -i);
    double relres = 0;
    int count = 0;

    Vector Sol =
        pcgPreconditioned(&A, &B, &zeros, eps, &relres, &count, &L, &Lt);

    double max = vectors_max_diff(&us, &Sol);
    printf("iter = %i, eps = %.15f, max_comp_diff =  %.15f\n", count, eps, max);

    free_vector(Sol);

    free_vector(x);
    free_vector(y);
    free_vector(us);
    free_vector(B);
    free_matrix(&A);
    free_matrix(&L);
    free_matrix(&Lt);
    free_vector(zeros);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }

    // Convert argument to integer
    NUM_THREADS = atoi(argv[1]);
    if(NUM_THREADS <= 0)
    {
        printf("Error: Number of threads must be positive\n");
        return 1;
    }

    // printf("threads: %i\n", NUM_THREADS);

    // Allocate memory for threads and thread_data
    threads = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    thread_args = (thread_data*)malloc(NUM_THREADS * sizeof(thread_data));

    if(!threads || !thread_args)
    {
        printf("Error: Memory allocation failed\n");
        free(threads);
        free(thread_args);
        return 1;
    }

    double a = 0;
    double b = 1.625;
    double c = 0;
    double d = 1.625;
    double h = 0.025;

    errByEpsPcgChol(a, b, c, d, h);

    return 0;
}