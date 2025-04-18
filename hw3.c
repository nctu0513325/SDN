#include "bfs.h"
#include <cstdlib>
#include <omp.h>
#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet* list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet* list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void top_down_step(Graph g, VertexSet* frontier, VertexSet* new_frontier, int* distances, int level)
{
    #pragma omp parallel
    {
        VertexSet local_new_frontier;
        vertex_set_init(&local_new_frontier, g->num_nodes);

        #pragma omp for
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];
                if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, level))
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
}

void bfs_top_down(Graph graph, solution* sol)
{
    VertexSet list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet* frontier = &list1;
    VertexSet* new_frontier = &list2;

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 1;
    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances, level);

        VertexSet* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        level++;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
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

        #pragma omp parallel for reduction(&&:done)
        for (int i = 0; i < graph->num_nodes; i++)
        {
            if (sol->distances[i] == NOT_VISITED_MARKER)
            {
                for (int j = graph->incoming_starts[i]; j < (i == graph->num_nodes - 1 ? graph->num_edges : graph->incoming_starts[i + 1]); j++)
                {
                    int incoming = graph->incoming_edges[j];
                    if (frontier[incoming])
                    {
                        sol->distances[i] = level;
                        new_frontier[i] = true;
                        done = false;
                        break;
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < graph->num_nodes; i++)
        {
            frontier[i] = new_frontier[i];
            new_frontier[i] = false;
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
            top_down_step(graph, frontier, new_frontier, sol->distances, level);

            if (new_frontier->count > 0)
                done = false;

            VertexSet* tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
        else
        {
            #pragma omp parallel for reduction(&&:done)
            for (int i = 0; i < graph->num_nodes; i++)
            {
                if (sol->distances[i] == NOT_VISITED_MARKER)
                {
                    for (int j = graph->incoming_starts[i]; j < (i == graph->num_nodes - 1 ? graph->num_edges : graph->incoming_starts[i + 1]); j++)
                    {
                        int incoming = graph->incoming_edges[j];
                        if (bottom_up_frontier[incoming])
                        {
                            sol->distances[i] = level;
                            bottom_up_new_frontier[i] = true;
                            done = false;
                            break;
                        }
                    }
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < graph->num_nodes; i++)
            {
                bottom_up_frontier[i] = bottom_up_new_frontier[i];
                bottom_up_new_frontier[i] = false;
            }

            vertex_set_clear(frontier);
            #pragma omp parallel
            {
                VertexSet local_frontier;
                vertex_set_init(&local_frontier, graph->num_nodes);

                #pragma omp for nowait
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
