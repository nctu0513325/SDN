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

void bottom_up_step(Graph g, std::vector<bool>& frontier, std::vector<bool>& new_frontier, int* distances, int level)
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
    std::vector<bool> frontier(graph->num_nodes, false);
    std::vector<bool> new_frontier(graph->num_nodes, false);

    // 初始化所有節點為未訪問
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // 設置根節點
    frontier[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 1;
    while (true)
    {
        bottom_up_step(graph, frontier, new_frontier, sol->distances, level);

        if (std::all_of(new_frontier.begin(), new_frontier.end(), [](bool v) { return !v; }))
            break;

        frontier.swap(new_frontier);
        std::fill(new_frontier.begin(), new_frontier.end(), false);
        level++;
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    VertexSet list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet* frontier = &list1;
    VertexSet* new_frontier = &list2;

    std::vector<bool> bottom_up_frontier(graph->num_nodes, false);
    std::vector<bool> bottom_up_new_frontier(graph->num_nodes, false);

    // 初始化所有節點為未訪問
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // 設置根節點
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    bottom_up_frontier[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 1;
    bool use_top_down = true;

    while (frontier->count != 0 || std::any_of(bottom_up_frontier.begin(), bottom_up_frontier.end(), [](bool v) { return v; }))
    {
        // 根據frontier的大小決定使用top-down還是bottom-up
        if (frontier->count < graph->num_nodes / 20)
            use_top_down = true;
        else if (frontier->count > graph->num_nodes / 5)
            use_top_down = false;

        if (use_top_down)
        {
            vertex_set_clear(new_frontier);
            top_down_step(graph, frontier, new_frontier, sol->distances);

            VertexSet* tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
        else
        {
            bottom_up_step(graph, bottom_up_frontier, bottom_up_new_frontier, sol->distances, level);
            bottom_up_frontier.swap(bottom_up_new_frontier);
            std::fill(bottom_up_new_frontier.begin(), bottom_up_new_frontier.end(), false);

            // 將bottom-up frontier轉換為top-down frontier
            vertex_set_clear(frontier);
            for (int i = 0; i < graph->num_nodes; i++)
            {
                if (bottom_up_frontier[i])
                {
                    frontier->vertices[frontier->count++] = i;
                }
            }
        }

        level++;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}
