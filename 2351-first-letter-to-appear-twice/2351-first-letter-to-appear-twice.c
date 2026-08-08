#include<string.h>
char repeatedCharacter(char* s) {
    
    int arr[26] = {0};
    int len = strlen(s);
    for(int i=0; i<len ; i++)
    {
        if(arr[s[i] - 'a']==0)
        {
            arr[s[i]-'a']++;
        }
        else{
            return s[i];
        }
    }
    return '-1';
}