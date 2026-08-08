/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* productExceptSelf(int* nums, int numsSize, int* returnSize) {
    
    *returnSize = numsSize;
    int *ans = (int*) malloc(numsSize * sizeof(int));
    int leftproduct =1;
    for(int i=0; i<numsSize; i++)
    {
        ans[i] = leftproduct;
        leftproduct *= nums[i];
    }
    int rightproduct = 1;
    for(int i=numsSize-1; i>=0 ; i--)
    {
        ans[i] *= rightproduct;
        rightproduct *= nums[i];
    }
    return ans;
}