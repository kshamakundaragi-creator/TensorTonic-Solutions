class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for i in range(len(nums)):
            needed = target - nums[i]
            if needed in hashmap:
                return [hashmap[needed], i]
            hashmap[nums[i]]= i