
int maxProfit(int* prices, int pricesSize) {
    int minprice=prices[0],profit=0,maxprofit=0;
    for(int i=1; i<pricesSize;i++)
    {
        if(minprice>prices[i])
        {
            minprice=prices[i];
        }
        profit=prices[i]-minprice;
        if(profit>maxprofit)
        {
            maxprofit=profit;
        }
    } 
    return maxprofit;

}