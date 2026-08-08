/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdlib.h>

int** insert(int** intervals, int intervalsSize, int* intervalsColSize,
             int* newInterval, int newIntervalSize,
             int* returnSize, int** returnColumnSizes)
{
    // Allocate memory
    int **answer = (int **)malloc((intervalsSize + 1) * sizeof(int *));
    *returnColumnSizes = (int *)malloc((intervalsSize + 1) * sizeof(int));

    int index = 0;
    int i = 0;

    // Phase 1: Copy intervals before the new interval
    while (i < intervalsSize && intervals[i][1] < newInterval[0])
    {
        answer[index] = (int *)malloc(2 * sizeof(int));

        answer[index][0] = intervals[i][0];
        answer[index][1] = intervals[i][1];

        (*returnColumnSizes)[index] = 2;

        index++;
        i++;
    }

    // Phase 2: Merge overlapping intervals
    while (i < intervalsSize && intervals[i][0] <= newInterval[1])
    {
        if (intervals[i][0] < newInterval[0])
            newInterval[0] = intervals[i][0];

        if (intervals[i][1] > newInterval[1])
            newInterval[1] = intervals[i][1];

        i++;
    }

    // Phase 3: Store the merged interval
    answer[index] = (int *)malloc(2 * sizeof(int));

    answer[index][0] = newInterval[0];
    answer[index][1] = newInterval[1];

    (*returnColumnSizes)[index] = 2;

    index++;

    // Phase 4: Copy remaining intervals
    while (i < intervalsSize)
    {
        answer[index] = (int *)malloc(2 * sizeof(int));

        answer[index][0] = intervals[i][0];
        answer[index][1] = intervals[i][1];

        (*returnColumnSizes)[index] = 2;

        index++;
        i++;
    }

    *returnSize = index;

    return answer;
}