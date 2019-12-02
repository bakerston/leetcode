def findMode(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """

    if not root:
        return []
    queue = [root]
    mydir = {}
    res = []

    while queue:
        node = queue.pop()
        mydir[node.val] = mydir.get(node.val, 0) + 1
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    maxfreq = max(mydir.values())
    for key, val in mydir.items():
        if val == maxfreq:
            res.append(key)
    return res