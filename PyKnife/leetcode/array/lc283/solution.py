# _*_ coding: utf-8 _*_

"""
This is the solution of no.283 problem in the LeetCode,
where the website of the problem is as follow:
https://leetcode.com/problems/move-zeroes/

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/12/07
"""


class Solution1:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        result = [0 for _ in range(len(nums))]
        i = 0
        for num in nums:
            if num != 0:
                result[i] = num
                i += 1
        for i in range(len(nums)):
            nums[i] = result[i]


class Solution2:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        index = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[index] = nums[i]
                index += 1

        while index < len(nums):
            nums[index] = 0
            index += 1


class Solution3:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1


def main():
    nums = [0, 1, 0, 3, 12]
    print(nums)
    (Solution3()).moveZeroes(nums)
    print(nums)


if __name__ == '__main__':
    main()
