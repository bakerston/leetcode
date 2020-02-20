class Solution:
    def hasPathSum(self, root, sum):
        if not root:
            return False
        elif root and root.left == None and root.right == None and root.val == sum:
            return True
        else:
            sum = sum - root.val
            if root.left and root.right == None:
                return self.hasPathSum(root.left, sum)
            elif root.right and root.left == None:
                return self.hasPathSum(root.right, sum)
            else:
                return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

