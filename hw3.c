#include "page_rank.h"
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include "../common/graph.h"

void page_rank(Graph g, double *solution, double damping, double convergence)
{
    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;

    // Initialize solution array
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
    }

    double *score_new = (double *)malloc(nnodes * sizeof(double));
    bool converged = false;
    double global_diff;

    // Identify nodes with no outgoing edges
    bool *no_outgoing = (bool *)calloc(nnodes, sizeof(bool));
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        if (outgoing_size(g, i) == 0)
        {
            no_outgoing[i] = true;
        }
    }

    while (!converged)
    {
        // Reset score_new and compute dangling sum
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 0; i < nnodes; ++i)
        {
            score_new[i] = 0.0;
            if (no_outgoing[i])
            {
                dangling_sum += solution[i];
            }
        }

        // Compute new scores
        #pragma omp parallel for
        for (int i = 0; i < nnodes; ++i)
        {
            const Vertex *in_begin = incoming_begin(g, i);
            const Vertex *in_end = incoming_end(g, i);
            for (const Vertex *v = in_begin; v != in_end; ++v)
            {
                score_new[i] += solution[*v] / outgoing_size(g, *v);
            }
            score_new[i] = damping * score_new[i] + (1.0 - damping) / nnodes;
        }

        // Add contribution from dangling nodes
        double dangling_contribution = damping * dangling_sum / nnodes;
        #pragma omp parallel for
        for (int i = 0; i < nnodes; ++i)
        {
            score_new[i] += dangling_contribution;
        }

        // Compute global difference and check for convergence
        global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff)
        for (int i = 0; i < nnodes; ++i)
        {
            global_diff += fabs(score_new[i] - solution[i]);
            solution[i] = score_new[i];
        }

        converged = (global_diff < convergence);
    }

    free(score_new);
    free(no_outgoing);
}
