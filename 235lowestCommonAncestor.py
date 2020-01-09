# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not self.isBothAncestor:
            return False

        while self.isBothAncestor(root, p, q) == True:
            leftNode = root.left
            rightNode = root.right
            return self.isBothAncestor(leftNode, p, q) or self.isBothAncestor(rightNode, p, q)
        return root

    def isBothAncestor(self, root, p, q):
        return self.isAncestor(root, p) and self.isAncestor(root, q)

    def isAncestor(self, root, p):
        if root == p:
            return True
        else:
            return None
        return self.isAncestor(root.left, p) or self.isAncestor(root.right, p)
