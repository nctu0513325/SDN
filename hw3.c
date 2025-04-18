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

            // Update bottom_up_frontier for consistency
            #pragma omp parallel for
            for (int i = 0; i < graph->num_nodes; i++)
                bottom_up_frontier[i] = (sol->distances[i] == level);

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

            // Update frontier for consistency
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
