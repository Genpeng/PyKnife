# _*_ coding: utf-8 _*_

"""
Implementation of insertion sort algorithm.

References:
[1] https://sort.hust.cc/3.insertionsort
[2] https://runestone.academy/runestone/books/published/pythonds/SortSearch/TheInsertionSort.html
[3] https://time.geekbang.org/column/article/41802

Author: Genpeng Xu
"""

from typing import List


class InsertionSort:
    @staticmethod
    def sort(arr: List[int]) -> None:
        for i in range(1, len(arr)):
            curr_elem = arr[i]
            j = i
            while j > 0 and curr_elem < arr[j-1]:
                arr[j] = arr[j-1]
                j -= 1
            arr[j] = curr_elem


if __name__ == '__main__':
    arr = [10, 7, 5, 3, 2, 4, 1, 8, 6, 9]
    print(arr)
    InsertionSort.sort(arr)
    print(arr)
