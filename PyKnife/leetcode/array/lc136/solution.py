# _*_ coding: utf-8 _*_

"""
This is the solution of no. 136 problem in the LeetCode,
where the website of the problem is as follow:
https://leetcode.com/problems/single-number/

The description of problem is as follow:
==========================================================================================================
Given a non-empty array of integers, every element appears twice except for one. Find that single one.

Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

Example 1:
Input: [2,2,1]
Output: 1

Example 2:
Input: [4,1,2,1,2]
Output: 4
==========================================================================================================

Author: StrongXGP (xgp1227@gmail.com)
Date:   2019/01/04
"""


class Solution1:
    def singleNumber(self, nums):
        """
        解法一：列表
        时间复杂度：O(n^2)
        空间复杂度：O(n)

        :type nums: List[int]
        :rtype: int
        """
        l = []
        for num in nums:
            if num in l:
                l.remove(num)
            else:
                l.append(num)
        return l.pop()


class Solution2:
    def singleNumber(self, nums):
        """
        解法二：哈希表
        时间复杂度：O(n)
        空间复杂度：O(n)

        :type nums: List[int]
        :rtype: int
        """
        d = dict()
        for num in nums:
            try:
                d.pop(num)
            except:
                d[num] = 1
        return d.popitem()[0]


class Solution3:
    def singleNumber(self, nums):
        """
        解法三：数学运算
        时间复杂度：O(n)
        空间复杂度：O(n)

        :type nums: List[int]
        :rtype: int
        """
        return 2 * sum(set(nums)) - sum(nums)


class Solution4:
    def singleNumber(self, nums):
        """
        解法四：位运算
        时间复杂度：O(n)
        空间复杂度：O(n)

        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for num in nums:
            res ^= num
        return res


def main():
    nums = [2, 2, 1]
    print((Solution4()).singleNumber(nums))


if __name__ == '__main__':
    main()
