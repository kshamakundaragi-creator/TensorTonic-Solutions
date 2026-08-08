#include<string.h>
char* removeDigit(char* number, char digit) {
    int len = strlen(number);
    int removeIndex =-1;
    for(int i=0; i<len; i++)
    {
        if(number[i]==digit)
        {
            removeIndex = i;
            if(i<len-1  && number[i+1]>number[i])
            {
                break;
            }
        }
    }
    char *ans = malloc(len * sizeof(char));
    int j=0;
    for(int i =0; i<len;i++)
    {
        if(i!= removeIndex)
        {
           ans[j]=number[i];
           j++;
        }
    }
    ans[j] = '\0';
    return ans;
}