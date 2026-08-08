#include <stdbool.h>

#define TABLE_SIZE 10

typedef struct
{
    int value;
    bool occupied;
} HashNode;

// Hash Function
int hashFunction(int value)
{
    return value % TABLE_SIZE;
}

// Search Function
bool search(HashNode hashTable[], int value)
{
    int pos = hashFunction(value);

    if (hashTable[pos].occupied == true &&
        hashTable[pos].value == value)
    {
        return true;
    }

    return false;
}

// Insert Function
void insert(HashNode hashTable[], int value)
{
    int pos = hashFunction(value);

    hashTable[pos].value = value;
    hashTable[pos].occupied = true;
}

bool isValidSudoku(char** board, int boardSize, int* boardColSize)
{
    // ---------- Check Rows ----------
    for (int i = 0; i < 9; i++)
    {
        HashNode hashTable[TABLE_SIZE];

        for (int k = 0; k < TABLE_SIZE; k++)
            hashTable[k].occupied = false;

        for (int j = 0; j < 9; j++)
        {
            if (board[i][j] == '.')
                continue;

            int value = board[i][j] - '0';

            if (search(hashTable, value))
                return false;

            insert(hashTable, value);
        }
    }

    // ---------- Check Columns ----------
    for (int i = 0; i < 9; i++)
    {
        HashNode hashTable[TABLE_SIZE];

        for (int k = 0; k < TABLE_SIZE; k++)
            hashTable[k].occupied = false;

        for (int j = 0; j < 9; j++)
        {
            if (board[j][i] == '.')
                continue;

            int value = board[j][i] - '0';

            if (search(hashTable, value))
                return false;

            insert(hashTable, value);
        }
    }

    // ---------- Check 3x3 Boxes ----------
    for (int row = 0; row < 9; row += 3)
    {
        for (int col = 0; col < 9; col += 3)
        {
            HashNode hashTable[TABLE_SIZE];

            for (int k = 0; k < TABLE_SIZE; k++)
                hashTable[k].occupied = false;

            for (int i = row; i < row + 3; i++)
            {
                for (int j = col; j < col + 3; j++)
                {
                    if (board[i][j] == '.')
                        continue;

                    int value = board[i][j] - '0';

                    if (search(hashTable, value))
                        return false;

                    insert(hashTable, value);
                }
            }
        }
    }

    return true;
}