void workerThreadStart(WorkerArgs *const args)
{
    int height = args->height;
    int width = args->width;
    int numThreads = args->numThreads;
    int threadId = args->threadId;

    // Calculate the number of rows each thread should process
    int rowsPerThread = height / numThreads;
    int remainingRows = height % numThreads;

    // Calculate the start row and number of rows for this thread
    int startRow = threadId * rowsPerThread;
    int numRows = rowsPerThread;

    // Distribute any remaining rows to the first few threads
    if (threadId < remainingRows) {
        startRow += threadId;
        numRows++;
    } else {
        startRow += remainingRows;
    }

    // Call mandelbrotSerial to compute this thread's portion of the image
    mandelbrotSerial(
        args->x0, args->y0, args->x1, args->y1,
        width, height,
        startRow, numRows,
        args->maxIterations,
        args->output
    );

    printf("Thread %d computed rows %d to %d\n", threadId, startRow, startRow + numRows - 1);
}

# requirement
void workerThreadStart(WorkerArgs *const args)
{
    int height = args->height;
    int width = args->width;
    int threadId = args->threadId;
    
    int startRow = (threadId == 0) ? 0 : height / 2;
    int numRows = height / 2;

    mandelbrotSerial(
        args->x0, args->y0, args->x1, args->y1,
        width, height,
        startRow, numRows,
        args->maxIterations,
        args->output
    );
}

void workerThreadStart(WorkerArgs *const args)
{
    int height = args->height;
    int width = args->width;
    int numThreads = args->numThreads;
    int threadId = args->threadId;

    int rowsPerThread = height / numThreads;
    int startRow = threadId * rowsPerThread;
    int numRows = (threadId == numThreads - 1) ? (height - startRow) : rowsPerThread;

    mandelbrotSerial(
        args->x0, args->y0, args->x1, args->y1,
        width, height,
        startRow, numRows,
        args->maxIterations,
        args->output
    );
}

void workerThreadStart(WorkerArgs *const args)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // ... 原有代码 ...

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Thread %d took %f seconds\n", args->threadId, diff.count());
}

void workerThreadStart(WorkerArgs *const args)
{
    int height = args->height;
    int width = args->width;
    int numThreads = args->numThreads;
    int threadId = args->threadId;

    for (int row = threadId; row < height; row += numThreads) {
        mandelbrotSerial(
            args->x0, args->y0, args->x1, args->y1,
            width, height,
            row, 1,
            args->maxIterations,
            args->output
        );
    }
}
