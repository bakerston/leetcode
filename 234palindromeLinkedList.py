#Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def isPalindrome(self, head):

        def isPalindrome(self, head):
            fast = slow = head
            # find the mid node
            while fast and fast.next:
                fast = fast.next.next
                slow = slow.next
            # reverse the second half
            node = None
            while slow:
                nxt = slow.next
                slow.next = node
                node = slow
                slow = nxt
            # compare the first and second half nodes
            while node:  # while node and head:
                if node.val != head.val:
                    return False
                node = node.next
                head = head.next
            return True