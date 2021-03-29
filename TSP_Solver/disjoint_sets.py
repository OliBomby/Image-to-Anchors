class disjoint_set:
	def __init__(self,vertex):
		self.parent = self
		self.rank = 0
		self.vertex = vertex
	def find(self):
		if self.parent != self:
			self.parent = self.parent.find()
		return self.parent
	def joinSets(self,otherTree):
		root = self.find()
		otherTreeRoot = otherTree.find()
		if root == otherTreeRoot:
			return
		if root.rank < otherTreeRoot.rank:
			root.parent = otherTreeRoot
		elif otherTreeRoot.rank < root.rank:
			otherTreeRoot.parent = root
		else:
			otherTreeRoot.parent = root
			root.rank += 1

