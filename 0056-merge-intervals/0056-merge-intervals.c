#include <stdlib.h>

// Comparator function for qsort
int compare(const void *a, const void *b)
{
    int *x = *(int **)a;
    int *y = *(int **)b;

    return x[0] - y[0];
}

int** merge(int** intervals, int intervalsSize, int* intervalsColSize,
            int* returnSize, int** returnColumnSizes)
{
    if(intervalsSize == 0)
    {
        *returnSize = 0;
        *returnColumnSizes = NULL;
        return NULL;
    }

    // Sort intervals based on starting point
    qsort(intervals, intervalsSize, sizeof(int *), compare);

    // Allocate memory for answer
    int **answer = (int **)malloc(intervalsSize * sizeof(int *));
    *returnColumnSizes = (int *)malloc(intervalsSize * sizeof(int));

    int index = 0;

    int currentStart = intervals[0][0];
    int currentEnd = intervals[0][1];

    for(int i = 1; i < intervalsSize; i++)
    {
        int nextStart = intervals[i][0];
        int nextEnd = intervals[i][1];

        // Overlap
        if(nextStart <= currentEnd)
        {
            if(nextEnd > currentEnd)
                currentEnd = nextEnd;
        }
        else
        {
            // Store current interval
            answer[index] = (int *)malloc(2 * sizeof(int));
            answer[index][0] = currentStart;
            answer[index][1] = currentEnd;
            (*returnColumnSizes)[index] = 2;
            index++;

            // Current becomes next
            currentStart = nextStart;
            currentEnd = nextEnd;
        }
    }

    // Store the last interval
    answer[index] = (int *)malloc(2 * sizeof(int));
    answer[index][0] = currentStart;
    answer[index][1] = currentEnd;
    (*returnColumnSizes)[index] = 2;
    index++;

    *returnSize = index;

    return answer;
}