import random
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


class Bounds:
    def __init__(self, nodeDict, adjMatrix):
        self.nodeDict = nodeDict
        self.adjMatrix = adjMatrix
        self.counts = len(nodeDict)
        self.edgeDict = {}
        for i in range(self.counts):
            for j in range(i + 1, self.counts):
                vertices = (i, j)
                self.edgeDict[vertices] = self.adjMatrix[i, j]

    '''
	Held-Karp Lower Bound 
       An iterative estimation that provides the tightest lower bound for a TSP. The HKLB differs based on U (Target Value).
	   One can determine the best HKLB through experimentations of U for each TSP instance.
	'''

    def calculateHKLB(self):
        # Input Parameters
        # U our upper bound target value is selected as roughly 115% of the OTB lower bound
        U = 1.15 * self.calculateOTB(self.adjMatrix)[0]
        iterationFactor = 0.015
        maxChanges = 100
        hkBound = -10000000000000000
        tsmall = 0.001
        alpha = 1
        beta = 0.5
        nodeNumbers = np.zeros(self.counts)
        numIterations = int(round(iterationFactor * self.counts))
        if numIterations == 0:
            numIterations += 1
        tVector = np.zeros(numIterations)
        newAdjMat = self.adjMatrix.copy()
        for i in range(0, maxChanges):
            for k in range(0, numIterations):
                # Calcuate the new edge weights based on nodeNumbers
                tempMatrix = self.calcNewMatrix(newAdjMat, nodeNumbers)
                result = self.calculateOptimalOTB(tempMatrix)
                oneTreeBound = result[0]
                oneTreeEdges = result[1]
                # HKBound is given as the sum of the OTB of the adjusted edges and 2* the sum of the nodeNumbers
                newHkBound = oneTreeBound + 2 * np.sum(nodeNumbers)
                # Improvement of hKBound
                if newHkBound > hkBound:
                    hkBound = newHkBound
                # aTour contains a boolean that says if it's a tour and corresponding degDict
                aTour = self.isATourOT(oneTreeEdges)
                if aTour[0]:
                    return hkBound
                degsVals = list(aTour[1].values())
                sumAllDegs = float(np.sum(np.square(2 - np.array(degsVals))))
                tVector[k] = alpha * (U - newHkBound) / sumAllDegs
                # Terminate when the stepSize is too small
                if tVector[k] < tsmall:
                    return hkBound
                deltNode = tVector[k] * (2 - np.array(degsVals))
                nodeNumbers = nodeNumbers + deltNode
            # Changes the decrement factor each loop
            alpha = beta * alpha
        return hkBound

    def calcNewMatrix(self, adjMatrix, nodeNumbers):
        temp = adjMatrix.copy()
        m = len(temp)
        # i is the index
        for i in range(0, m):
            temp[i] -= nodeNumbers[i]
            temp[:, i] -= nodeNumbers[i]
            temp[i][i] = 0
        return temp

    # This function only checks if each node in the 1-tree has degree 2. A 1-tree implies connectedness. If every node has degree 2,
    # a one-tree must be a tour.
    def isATourOT(self, oneTree):
        nodes = range(0, self.counts)
        degreeDict = {node: 0 for node in nodes}
        for edge in oneTree:
            x = edge[0]
            y = edge[1]
            degreeDict[x] += 1
            degreeDict[y] += 1
        for i in nodes:
            if degreeDict[i] != 2:
                return [False, degreeDict]
        return [True, degreeDict]

    '''
	1-tree Bound 
	  	A form of lower bound that utilizes the 1-tree based on Chapter 7 of The Traveling Salesman Problem: A Computational Study by Cook
			1. Pick a random node v0.
 			2. Get the length of the MST after disregarding the random node. 
 			3. Let S be the sum of the cheapest two edges incident with the random node v0. 
 			4. Output the sum of 2 and 3.
 		The 1-Tree bound should approximately be 90.5% of the optimal cost. The best 1-Tree lower bound will be the maximum cost of the many MSTs we get.
 	'''

    def calculateOTB(self, adjMatrix):
        maxOTBLB = -10000000
        bestTree = []
        for initNode in range(0, self.counts):
            MSTedges = self.OTBHelper(adjMatrix, initNode)
            r = self.calcCost(MSTedges)
            # s is the sum of the cheapest two edges incident with the random node v0.
            s = 0
            edgeLengths = adjMatrix[initNode]
            nodeNums = range(0, self.counts)
            twoNN = sorted(zip(edgeLengths, nodeNums))[1:3]
            s = twoNN[0][0] + twoNN[1][0]
            temp = r + s
            if temp > maxOTBLB:
                maxOTBLB = temp
                oneTreeEdges = MSTedges[:]
                oneTreeEdges.append((initNode, twoNN[0][1]))
                oneTreeEdges.append((initNode, twoNN[1][1]))
                bestTree = oneTreeEdges
        return [maxOTBLB, bestTree]

    def calculateOptimalOTB(self, adjMatrix):
        minOTBLB = 1000000
        bestTree = []
        for initNode in range(0, self.counts):
            MSTedges = self.OTBHelper(adjMatrix, initNode)
            r = self.calcAdjustedCost(MSTedges, adjMatrix)
            # s is the sum of the cheapest two edges incident with the random node v0.
            s = 0
            edgeLengths = adjMatrix[initNode]
            nodeNums = range(0, self.counts)
            twoNN = sorted(zip(edgeLengths, nodeNums))[1:3]
            s = twoNN[0][0] + twoNN[1][0]
            temp = r + s
            if temp < minOTBLB:
                minOTBLB = temp
                oneTreeEdges = MSTedges[:]
                oneTreeEdges.append((initNode, twoNN[0][1]))
                oneTreeEdges.append((initNode, twoNN[1][1]))
                bestTree = oneTreeEdges
        return [minOTBLB, bestTree]

    def OTBHelper(self, adjMatrix, initNode):
        # Create an AdjMatrix without the row & col containing the initNode
        newAdjMat = adjMatrix
        newAdjMat = np.delete(newAdjMat, initNode, axis=0)
        newAdjMat = np.delete(newAdjMat, initNode, axis=1)
        # Calculate MST length without the initNode
        mst = minimum_spanning_tree(newAdjMat)
        MSTedges = []
        Z = mst.toarray().astype(float)
        for i in range(len(Z)):
            array = np.nonzero(Z[i])[0]
            for index in array:
                x = i
                y = index
                if i >= initNode:
                    x += 1
                if index >= initNode:
                    y += 1
                tuplex = (x, y)
                MSTedges.append(tuplex)
        return MSTedges

    def calcAdjustedCost(self, MSTedges, adjMatrix):
        r = 0
        for edge in MSTedges:
            r += adjMatrix[edge[0], edge[1]]
        return r

    def calcCost(self, MSTedges):
        # r is the length of the MST we have without the initNode
        r = 0
        for edge in MSTedges:
            checkEdge = edge
            if (checkEdge not in self.edgeDict):
                checkEdge = (edge[1], edge[0])
            r += self.edgeDict[checkEdge]
        return r

    '''
	  MST Upper Bound 
	    Simply 2* the MST cost of the original dataSet
	'''

    def calculateMSTUpperBound(self):
        mst = minimum_spanning_tree(self.adjMatrix)
        MSTedges = []
        Z = mst.toarray().astype(float)
        for i in range(len(Z)):
            array = np.nonzero(Z[i])[0]
            for index in array:
                tuplex = (i, index)
                MSTedges.append(tuplex)
        cost = 0
        for edge in MSTedges:
            checkEdge = edge
            if (checkEdge not in self.edgeDict):
                checkEdge = (edge[1], edge[0])
            cost += self.edgeDict[checkEdge]
        return 2 * cost
