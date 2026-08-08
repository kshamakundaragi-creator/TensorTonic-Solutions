#include<string.h>
int strStr(char* haystack, char* needle) {
  
     int n = strlen(needle);
     int m = strlen(haystack);
       if(n== 0)
        return 0;
    
    if(n > m )
    return -1;

  for(int i=0; i<=m-n ;i++)
  { 
     int j=0;
     while(j<n && haystack[i+j]== needle[j])
     {
        j++;
     }
     if(j==n)
     return i;
  }
  return -1;
    
    
}