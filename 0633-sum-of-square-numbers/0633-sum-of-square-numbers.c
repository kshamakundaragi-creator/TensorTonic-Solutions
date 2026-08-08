#include<math.h>
bool judgeSquareSum(int c) {
    if(c<0)
    {
        return false;
    }
     long long a=0; 
     long long b= (long long)sqrt(c);
     while(a<=b)
     {
        long long current_sum = (a*a) + (b*b);
        if(current_sum ==c)
        {
            return true;
        }
        else if(current_sum <c)
        {
            a++;
        }
        else{
            b--;
        }
     }
    return false;
}