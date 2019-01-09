# _*_ coding: utf-8 _*_

"""
This is the solution of no. 349 problem in the LeetCode,
where the website of the problem is as follow:
https://leetcode.com/problems/intersection-of-two-arrays/

The description of problem is as follow:
==========================================================================================================
Given two arrays, write a function to compute their intersection.

Example 1:
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]

Example 2:
Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [9,4]

Note:
Each element in the result must be unique.
The result can be in any order.
==========================================================================================================

Author: StrongXGP (xgp1227@gmail.com)
Date:   2019/01/09
"""


class Solution1:
    def intersection(self, nums1, nums2):
        """
        解法1：使用两个集合
        时间复杂度：O(n)
        空间复杂度：O(n)

        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if nums1 is None or nums2 is None:
            raise Exception("[ERROR] The exists at least one null array!")

        # return list(set(nums1) & set(nums2))

        s1, s2, res = set(nums1), set(nums2), list()
        for num in s1:
            if num in s2:
                res.append(num)
        return res


class Solution2:
    def intersection(self, nums1, nums2):
        """
        解法2：双指针
        时间复杂度：O(n * log(n))
        空间复杂度：O(k)，其中，k表示交集的大小

        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if nums1 is None or nums2 is None:
            raise Exception("[ERROR] The exists at least one null array!")

        nums1, nums2 = sorted(nums1), sorted(nums2)

        i, j, s = 0, 0, set()
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                s.add(nums1[i])
                i += 1
                j += 1
        return list(s)


class Solution3:
    def intersection(self, nums1, nums2):
        """
        解法3：二分查找
        时间复杂度：O(n * log(n))
        空间复杂度：O(k)，其中，k表示交集的大小

        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        def binary_search(nums, target):
            low, high = 0, len(nums) - 1
            while low <= high:
                mid = low + (high - low) // 2
                if nums[mid] < target:
                    low = mid + 1
                elif nums[mid] > target:
                    high = mid - 1
                else:
                    return True
            return False

        if nums1 is None or nums2 is None:
            raise Exception("[ERROR] The exists at least one null array!")

        if len(nums1) == 0 or len(nums2) == 0:
            return []

        nums2 = sorted(nums2)

        s = set()
        for num in nums1:
            if binary_search(nums2, num):
                s.add(num)
        return list(s)


def main():
    nums1 = [1, 2, 2, 1]
    nums2 = [2, 2]
    print((Solution3()).intersection(nums1, nums2))


if __name__ == '__main__':
    main()
