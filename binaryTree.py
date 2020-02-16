class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.leftSym(root)==self.rightSym(root)

    def leftSym(self,root):
        alist=[]
        if root==None:
            alist.append('x')
            return alist
        alist.append(root.val)
        alist+=self.leftSym(root.left)
        alist+=self.leftSym(root.right)
        return alist
    def rightSym(self,root):
        alist=[]
        if root==None:
            alist.append('x')
            return alist
        alist.append(root.val)
        alist+=self.rightSym(root.right)
        alist+=self.rightSym(root.left)
        return alist

