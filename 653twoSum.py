def findTarget(self, root, k):
    """
    :type root: TreeNode
    :type k: int
    :rtype: bool
    """
    if not root:
        return False

    return self._findTarget(root, set(), k)


def _findTarget(self, node, nodes, k):
    if not node:
        return False

    complement = k - node.val
    if complement in nodes:
        return True

    nodes.add(node.val)

    return self._findTarget(node.left, nodes, k) or self._findTarget(node.right, nodes, k)