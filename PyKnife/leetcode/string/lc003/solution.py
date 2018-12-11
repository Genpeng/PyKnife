# _*_ coding: utf-8 _*_

"""
This is the solution of no. 003 problem in the LeetCode,
where the website of the problem is as follow:
https://leetcode.com/problems/longest-substring-without-repeating-characters/

The description of problem is as follow:
==========================================================================================================
Given a string, find the length of the longest substring without repeating characters.

Example 1:

Input: "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
==========================================================================================================

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/12/11
"""


class Solution1:
    def lengthOfLongestSubstring(self, s):
        """
        方法1：暴力枚举法（Time Limit Exceeded）

        :type s: str
        :rtype: int
        """
        def is_unique(s, start, end):
            ch_has_seen = set()
            for c in s[start:end]:
                if c in ch_has_seen:
                    return False
                else:
                    ch_has_seen.add(c)
            return True

        max_len = 0
        for l in range(len(s)):
            for r in range(1, len(s) + 1):
                if is_unique(s, l, r):
                    max_len = max(r - l, max_len)
        return max_len


class Solution2:
    def lengthOfLongestSubstring(self, s):
        """
        方法2：滑动窗口
        执行用时：152 ms
        已经战胜 43.27 % 的 python3 提交记录

        :type s: str
        :rtype: int
        """
        l, r, max_len = 0, 0, 0
        substring = set()
        while l < len(s) and r < len(s):
            if s[r] in substring:
                substring.remove(s[l])
                l += 1
            else:
                substring.add(s[r])
                r += 1
                max_len = max(r - l, max_len)
        return max_len


class Solution3:
    def lengthOfLongestSubstring(self, s):
        """
        方法3：滑动窗口（优化）
        执行用时：88 ms
        已经战胜 99.02 % 的 python3 提交记录

        :type s: str
        :rtype: int
        """
        # l, max_len = 0, 0
        # chars_has_seen = dict()
        # for r, char in enumerate(s):
        #     if char in chars_has_seen:
        #         l = max(chars_has_seen[char], l)
        #     max_len = max(r - l + 1, max_len)
        #     chars_has_seen[char] = r + 1
        # return max_len

        l, max_len = 0, 0
        chars_has_seen = dict()
        for r, char in enumerate(s):
            if char in chars_has_seen and l <= chars_has_seen[char]:
                l = chars_has_seen[char] + 1
            else:
                max_len = max(r - l + 1, max_len)
            chars_has_seen[char] = r
        return max_len


class Solution4:
    def lengthOfLongestSubstring(self, s):
        """
        方法4：滑动窗口（已知字符集）
        执行用时：124 ms
        已经战胜 68.97 % 的 python3 提交记录

        :type s: str
        :rtype: int
        """
        all_chars = [0 for _ in range(128)]
        l, max_len = 0, 0
        for r, char in enumerate(s):
            l = max(all_chars[ord(char)], l)  # 得到窗口的左边界
            max_len = max(r - l + 1, max_len)  # 保留最长长度
            all_chars[ord(char)] = r + 1  # 添加/更新字符的位置
        return max_len


def main():
    # s = "abba"
    s = "abcabcbb"
    print((Solution1()).lengthOfLongestSubstring(s))
    print((Solution2()).lengthOfLongestSubstring(s))
    print((Solution3()).lengthOfLongestSubstring(s))
    print((Solution4()).lengthOfLongestSubstring(s))


if __name__ == '__main__':
    main()
