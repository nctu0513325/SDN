#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "cg_impl.h"
#include "randdp.h"

void conj_grad(const int colidx[],
               const int rowstr[],
               const double x[],
               double z[],
               const double a[],
               double p[],
               double q[],
               double r[],
               double *rnorm)
{
    int cgit, cgitmax = 25;
    double d, sum, rho, rho0, alpha, beta;

    rho = 0.0;

    #pragma omp parallel
    {
        #pragma omp for
        for (int j = 0; j < naa + 1; j++)
        {
            q[j] = 0.0;
            z[j] = 0.0;
            r[j] = x[j];
            p[j] = r[j];
        }

        #pragma omp for reduction(+:rho)
        for (int j = 0; j < lastcol - firstcol + 1; j++)
        {
            rho += r[j] * r[j];
        }
    }

    for (cgit = 1; cgit <= cgitmax; cgit++)
    {
        #pragma omp parallel
        {
            #pragma omp for private(sum)
            for (int j = 0; j < lastrow - firstrow + 1; j++)
            {
                sum = 0.0;
                for (int k = rowstr[j]; k < rowstr[j + 1]; k++)
                {
                    sum += a[k] * p[colidx[k]];
                }
                q[j] = sum;
            }

            #pragma omp single
            {
                d = 0.0;
            }

            #pragma omp for reduction(+:d)
            for (int j = 0; j < lastcol - firstcol + 1; j++)
            {
                d += p[j] * q[j];
            }

            #pragma omp single
            {
                alpha = rho / d;
                rho0 = rho;
                rho = 0.0;
            }

            #pragma omp for reduction(+:rho)
            for (int j = 0; j < lastcol - firstcol + 1; j++)
            {
                z[j] += alpha * p[j];
                r[j] -= alpha * q[j];
                rho += r[j] * r[j];
            }

            #pragma omp single
            {
                beta = rho / rho0;
            }

            #pragma omp for
            for (int j = 0; j < lastcol - firstcol + 1; j++)
            {
                p[j] = r[j] + beta * p[j];
            }
        }
    }

    sum = 0.0;
    #pragma omp parallel
    {
        #pragma omp for private(d)
        for (int j = 0; j < lastrow - firstrow + 1; j++)
        {
            d = 0.0;
            for (int k = rowstr[j]; k < rowstr[j + 1]; k++)
            {
                d += a[k] * z[colidx[k]];
            }
            r[j] = d;
        }

        #pragma omp for reduction(+:sum) private(d)
        for (int j = 0; j < lastcol - firstcol + 1; j++)
        {
            d = x[j] - r[j];
            sum += d * d;
        }
    }

    *rnorm = sqrt(sum);
}

void makea(int n,
           int nz,
           double a[],
           int colidx[],
           int rowstr[],
           int firstrow,
           int lastrow,
           int firstcol,
           int lastcol,
           int arow[],
           int acol[][NONZER + 1],
           double aelt[][NONZER + 1],
           int iv[])
{
    int nzv, nn1;
    int ivc[NONZER + 1];
    double vc[NONZER + 1];

    nn1 = 1;
    do
    {
        nn1 = 2 * nn1;
    } while (nn1 < n);

    for (int iouter = 0; iouter < n; iouter++)
    {
        nzv = NONZER;
        sprnvc(n, nzv, nn1, vc, ivc);
        vecset(n, vc, ivc, &nzv, iouter + 1, 0.5);
        arow[iouter] = nzv;

        for (int ivelt = 0; ivelt < nzv; ivelt++)
        {
            acol[iouter][ivelt] = ivc[ivelt] - 1;
            aelt[iouter][ivelt] = vc[ivelt];
        }
    }

    sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, aelt, firstrow, lastrow, iv, RCOND, SHIFT);
}

void sparse(double a[],
            int colidx[],
            int rowstr[],
            int n,
            int nz,
            int nozer,
            const int arow[],
            int acol[][NONZER + 1],
            double aelt[][NONZER + 1],
            int firstrow,
            int lastrow,
            int nzloc[],
            double rcond,
            double shift)
{
    int nza, nzrow, jcol;
    double size, scale, ratio, va;
    bool cont40;

    const int nrows = lastrow - firstrow + 1;

    for (int j = 0; j < nrows + 1; j++)
    {
        rowstr[j] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (nza = 0; nza < arow[i]; nza++)
        {
            int j = acol[i][nza] + 1;
            rowstr[j] = rowstr[j] + arow[i];
        }
    }

    rowstr[0] = 0;
    for (int j = 1; j < nrows + 1; j++)
    {
        rowstr[j] = rowstr[j] + rowstr[j - 1];
    }
    nza = rowstr[nrows] - 1;

    if (nza > nz)
    {
        printf("Space for matrix elements exceeded in sparse\n");
        printf("nza, nzmax = %d, %d\n", nza, nz);
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < nrows; j++)
    {
        for (int k = rowstr[j]; k < rowstr[j + 1]; k++)
        {
            a[k] = 0.0;
            colidx[k] = -1;
        }
        nzloc[j] = 0;
    }

    size = 1.0;
    ratio = pow(rcond, (1.0 / (double)(n)));

    for (int i = 0; i < n; i++)
    {
        for (nza = 0; nza < arow[i]; nza++)
        {
            int j = acol[i][nza];

            scale = size * aelt[i][nza];
            for (nzrow = 0; nzrow < arow[i]; nzrow++)
            {
                int jcol = acol[i][nzrow];
                va = aelt[i][nzrow] * scale;

                if (jcol == j && j == i)
                {
                    va = va + rcond - shift;
                }

                cont40 = false;
                int k = rowstr[j];
                for (; k < rowstr[j + 1]; k++)
                {
                    if (colidx[k] > jcol)
                    {
                        for (int kk = rowstr[j + 1] - 2; kk >= k; kk--)
                        {
                            if (colidx[kk] > -1)
                            {
                                a[kk + 1] = a[kk];
                                colidx[kk + 1] = colidx[kk];
                            }
                        }
                        colidx[k] = jcol;
                        a[k] = 0.0;
                        cont40 = true;
                        break;
                    }
                    if (colidx[k] == -1)
                    {
                        colidx[k] = jcol;
                        cont40 = true;
                        break;
                    }
                    if (colidx[k] == jcol)
                    {
                        nzloc[j] = nzloc[j] + 1;
                        cont40 = true;
                        break;
                    }
                }
                if (cont40 == false)
                {
                    printf("internal error in sparse: i=%d\n", i);
                    exit(EXIT_FAILURE);
                }
                a[k] = a[k] + va;
            }
        }
        size = size * ratio;
    }

    for (int j = 1; j < nrows; j++)
    {
        nzloc[j] = nzloc[j] + nzloc[j - 1];
    }

    for (int j = 0; j < nrows; j++)
    {
        int j1 = 0;
        if (j > 0)
        {
            j1 = rowstr[j] - nzloc[j - 1];
        }
        else
        {
            j1 = 0;
        }
        int j2 = rowstr[j + 1] - nzloc[j];
        nza = rowstr[j];
        for (int k = j1; k < j2; k++)
        {
            a[k] = a[nza];
            colidx[k] = colidx[nza];
            nza = nza + 1;
        }
    }
    for (int j = 1; j < nrows + 1; j++)
    {
        rowstr[j] = rowstr[j] - nzloc[j - 1];
    }
}

void sprnvc(int n, int nz, int nn1, double v[], int iv[])
{
    double vecelt, vecloc;

    int nzv = 0;

    while (nzv < nz)
    {
        vecelt = randlc(&tran, amult);

        vecloc = randlc(&tran, amult);
        int i = icnvrt(vecloc, nn1) + 1;
        if (i > n)
            continue;

        bool was_gen = false;
        for (int ii = 0; ii < nzv; ii++)
        {
            if (iv[ii] == i)
            {
                was_gen = true;
                break;
            }
        }
        if (was_gen)
            continue;
        v[nzv] = vecelt;
        iv[nzv] = i;
        nzv = nzv + 1;
    }
}

int icnvrt(double x, int ipwr2)
{
    return (int)(ipwr2 * x);
}

void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
    bool set = false;
    for (int k = 0; k < *nzv; k++)
    {
        if (iv[k] == i)
        {
            v[k] = val;
            set = true;
        }
    }
    if (set == false)
    {
        v[*nzv] = val;
        iv[*nzv] = i;
        *nzv = *nzv + 1;
    }
}

void init(double *zeta)
{
    firstrow = 0;
    lastrow = NA - 1;
    firstcol = 0;
    lastcol = NA - 1;

    naa = NA;
    nzz = NZ;

    tran = 314159265.0;
    amult = 1220703125.0;
    *zeta = randlc(&tran, amult);

    makea(naa, nzz, a, colidx, rowstr, firstrow, lastrow, firstcol, lastcol, arow,
          (int(*)[NONZER + 1]) acol, (double(*)[NONZER + 1]) aelt, iv);

    for (int j = 0; j < lastrow - firstrow + 1; j++)
    {
        for (int k = rowstr[j]; k < rowstr[j + 1]; k++)
        {
            colidx[k] = colidx[k] - firstcol;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < NA + 1; i++)
    {
        x[i] = 1.0;
    }

    #pragma omp parallel for
    for (int j = 0; j < lastcol - firstcol + 1; j++)
    {
        q[j] = 0.0;
        z[j] = 0.0;
        r[j] = 0.0;
        p[j] = 0.0;
    }
}

void iterate(double *zeta, const int *it)
{
    double rnorm;
    double norm_temp1 = 0.0, norm_temp2 = 0.0;

    conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);

    #pragma omp parallel for reduction(+:norm_temp1,norm_temp2)
    for (int j = 0; j < lastcol - firstcol + 1; j++)
    {
        norm_temp1 += x[j] * z[j];
        norm_temp2 += z[j] * z[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    *zeta = SHIFT + 1.0 / norm_temp1;
    if (*it == 1)
        printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", *it, rnorm, *zeta);

    #pragma omp parallel for
    for (int j = 0; j < lastcol - firstcol + 1; j++)
    {
        x[j] = norm_temp2 * z[j];
    }
}
