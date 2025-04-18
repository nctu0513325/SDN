#include "bfs.h"
#include <cstdlib>
#include <omp.h>
#include <vector>
#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

// 原有的top-down相關函數保持不變

void bottom_up_step(Graph g, bool* frontier, bool* new_frontier, int* distances, int level)
{
    #pragma omp parallel for
    for (int i = 0; i < g->num_nodes; i++)
    {
        if (distances[i] == NOT_VISITED_MARKER)
        {
            for (int j = g->incoming_starts[i]; j < (i == g->num_nodes - 1 ? g->num_edges : g->incoming_starts[i + 1]); j++)
            {
                int incoming = g->incoming_edges[j];
                if (frontier[incoming])
                {
                    distances[i] = level;
                    new_frontier[i] = true;
                    break;
                }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    bool* frontier = new bool[graph->num_nodes]();
    bool* new_frontier = new bool[graph->num_nodes]();

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 1;
    bool done = false;

    while (!done)
    {
        done = true;
        bottom_up_step(graph, frontier, new_frontier, sol->distances, level);

        #pragma omp parallel for reduction(&&:done)
        for (int i = 0; i < graph->num_nodes; i++)
        {
            if (new_frontier[i])
            {
                done = false;
                frontier[i] = true;
                new_frontier[i] = false;
            }
        }
        level++;
    }

    delete[] frontier;
    delete[] new_frontier;
}

void bfs_hybrid(Graph graph, solution* sol)
{
    VertexSet list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet* frontier = &list1;
    VertexSet* new_frontier = &list2;

    bool* bottom_up_frontier = new bool[graph->num_nodes]();
    bool* bottom_up_new_frontier = new bool[graph->num_nodes]();

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    bottom_up_frontier[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 1;
    bool use_top_down = true;
    bool done = false;

    while (!done)
    {
        done = true;

        if (frontier->count < graph->num_nodes / 20)
            use_top_down = true;
        else if (frontier->count > graph->num_nodes / 5)
            use_top_down = false;

        if (use_top_down)
        {
            vertex_set_clear(new_frontier);
            #pragma omp parallel
            {
                VertexSet local_new_frontier;
                vertex_set_init(&local_new_frontier, graph->num_nodes);

                #pragma omp for
                for (int i = 0; i < frontier->count; i++)
                {
                    int node = frontier->vertices[i];
                    int start_edge = graph->outgoing_starts[node];
                    int end_edge = (node == graph->num_nodes - 1) ? graph->num_edges : graph->outgoing_starts[node + 1];

                    for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
                    {
                        int outgoing = graph->outgoing_edges[neighbor];
                        if (__sync_bool_compare_and_swap(&sol->distances[outgoing], NOT_VISITED_MARKER, level))
                        {
                            local_new_frontier.vertices[local_new_frontier.count++] = outgoing;
                        }
                    }
                }

                #pragma omp critical
                {
                    for (int i = 0; i < local_new_frontier.count; i++)
                    {
                        new_frontier->vertices[new_frontier->count++] = local_new_frontier.vertices[i];
                    }
                }

                vertex_set_destroy(&local_new_frontier);
            }

            if (new_frontier->count > 0)
                done = false;

            VertexSet* tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
        else
        {
            bottom_up_step(graph, bottom_up_frontier, bottom_up_new_frontier, sol->distances, level);

            #pragma omp parallel for reduction(&&:done)
            for (int i = 0; i < graph->num_nodes; i++)
            {
                if (bottom_up_new_frontier[i])
                {
                    done = false;
                    bottom_up_frontier[i] = true;
                    bottom_up_new_frontier[i] = false;
                }
            }

            vertex_set_clear(frontier);
            #pragma omp parallel
            {
                VertexSet local_frontier;
                vertex_set_init(&local_frontier, graph->num_nodes);

                #pragma omp for
                for (int i = 0; i < graph->num_nodes; i++)
                {
                    if (bottom_up_frontier[i])
                    {
                        local_frontier.vertices[local_frontier.count++] = i;
                    }
                }

                #pragma omp critical
                {
                    for (int i = 0; i < local_frontier.count; i++)
                    {
                        frontier->vertices[frontier->count++] = local_frontier.vertices[i];
                    }
                }

                vertex_set_destroy(&local_frontier);
            }
        }

        level++;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
    delete[] bottom_up_frontier;
    delete[] bottom_up_new_frontier;
}
