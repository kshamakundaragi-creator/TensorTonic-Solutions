int max(int a, int b, int c)
{
    int maximum=a;
    if(b>maximum)
    maximum=b;
    if(c>maximum)
    maximum =c;

  return maximum;
}
int min(int a, int b, int c)
{
    int minimum=a;
    if(b<minimum)
    minimum=b;
    if(c<minimum)
    minimum =c;

  return minimum;
}

int maxProduct(int* nums, int numsSize)
{
   int currentmax=nums[0];
   int currentmin=nums[0];
   int answer = nums[0];
   for(int i=1; i<numsSize; i++)
   {
    int tempmax = currentmax;
    int tempmin = currentmin;
    currentmax = max(tempmax*nums[i],tempmin*nums[i],nums[i]);
    currentmin = min(tempmax*nums[i],tempmin*nums[i],nums[i]);
    if(currentmax>answer)
    {
        answer=currentmax;
    }

   }
   return answer;
}