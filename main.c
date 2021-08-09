#include <stdio.h>
#include "spkmeans.h"

# define N 4

void r8vec_print(int n, double a[], char *title);
void r8mat_print_some(int m, int n, double a[], int ilo, int jlo, int ihi, int jhi, char *title);
void r8mat_print(int m, int n, double a[], char *title);

int main() {
    int i;
    double a[N * N] = {
            4.0, -30.0, 60.0, -35.0,
            -30.0, 300.0, -675.0, 420.0,
            60.0, -675.0, 1620.0, -1050.0,
            -35.0, 420.0, -1050.0, 700.0};
    int n = N;
    double d[N];
    double v[N * N] = {{0}};

    printf("\n");
    printf("TEST01\n");
    printf("  For a symmetric matrix A,\n");
    printf("  JACOBI_EIGENVALUE computes the eigenvalues D\n");
    printf("  and eigenvectors V so that A * V = D * V.\n");

    r8mat_print(n, n, a, "  Input matrix A:");

    jacobiAlgorithm(a, v, n);

    printf("\n");

    for (i = 0; i < n; i++)
        d[i] = a[i + i * n];
    r8vec_print(n, d, "  Eigenvalues D:");

    r8mat_print(n, n, v, "  Eigenvector matrix V:");

    printf("%d", EigengapHeuristicKCalc (a, n));
}

void r8vec_print(int n, double a[], char *title) {
    int i;

    fprintf(stdout, "\n");
    fprintf(stdout, "%s\n", title);
    fprintf(stdout, "\n");
    for (i = 0; i < n; i++) {
        fprintf(stdout, "  %8d: %14f\n", i, a[i]);
    }
}


void r8mat_print(int m, int n, double a[], char *title) {
    r8mat_print_some(m, n, a, 1, 1, m, n, title);
}

void r8mat_print_some(int m, int n, double a[], int ilo, int jlo, int ihi, int jhi, char *title) {
# define INCX 5

    int i;
    int i2hi;
    int i2lo;
    int j;
    int j2hi;
    int j2lo;

    fprintf(stdout, "\n");
    fprintf(stdout, "%s\n", title);

    if (m <= 0 || n <= 0) {
        fprintf(stdout, "\n");
        fprintf(stdout, "  (None)\n");
        return;
    }
    /*
      Print the columns of the matrix, in strips of 5.
    */
    for (j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX) {
        j2hi = j2lo + INCX - 1;
        if (n < j2hi) {
            j2hi = n;
        }
        if (jhi < j2hi) {
            j2hi = jhi;
        }

        fprintf(stdout, "\n");
        /*
          For each column J in the current range...

          Write the header.
        */
        fprintf(stdout, "  Col:  ");
        for (j = j2lo; j <= j2hi; j++) {
            fprintf(stdout, "  %7d     ", j - 1);
        }
        fprintf(stdout, "\n");
        fprintf(stdout, "  Row\n");
        fprintf(stdout, "\n");
        /*
          Determine the range of the rows in this strip.
        */
        if (1 < ilo) {
            i2lo = ilo;
        } else {
            i2lo = 1;
        }
        if (m < ihi) {
            i2hi = m;
        } else {
            i2hi = ihi;
        }

        for (i = i2lo; i <= i2hi; i++) {
            /*
              Print out (up to) 5 entries in row I, that lie in the current strip.
            */
            fprintf(stdout, "%5d:", i - 1);
            for (j = j2lo; j <= j2hi; j++) {
                fprintf(stdout, "  %14f", a[j - 1 + (i - 1) * m]);
            }
            fprintf(stdout, "\n");
        }
    }
# undef INCX
}

