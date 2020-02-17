class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root):
        if not root:
            return 0
        elif (not root.left) and (not root.right):
            return 1
        else:
            return 1+max(maxDepth(root.right),maxDepth(root.right))
