import monkdata as m
import dtree as d
import matplotlib.pyplot as p

# variables
partitions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
runs = 10
bigList1 = []
errorList1 = []
varianceList1 = []
bigList3 = []
errorList3 = []
varianceList3 = []

# getting the best tree from validationset
def bestPrunedTree(trainer, validation):
    max = 0
    pruneWays = d.allPruned(trainer)
    for tree in pruneWays:
        current = d.check(tree, validation)
        if (len(pruneWays) == 0):
            print("Prune completed, no more left.")
        if current > max:
            max = current
            max_tree = tree
    return max_tree

# compute variance calculation
def variance(all, mean):
    sum = 0
    for i in all:
        sum += (mean - float(i))**2
    spread = float(sum)/len(all)
    return spread

# determine the optimal partition of training/ validation by varying the parameter partition
def optimisePartitions1(): #runs
    tree1 = d.buildTree(m.monk1, m.attributes)
    score1 = d.check(tree1, m.monk1test)
    print("Performance of monk1 tree: " + str(score1) + "\n")
    for index, partition in enumerate(partitions):
        for j in range(runs):
            train1, val3 = d.partition(m.monk1, partition)
            tree1a = d.buildTree(train1, m.attributes)
            best1 = bestPrunedTree(tree1a, val3)
            bigList1.append(1 - d.check(best1, m.monk1test))
        errorList1.append(sum(bigList1) / len(bigList1))
        varianceList1.append(variance(bigList1, errorList1[index]))

    return errorList1, varianceList1

def optimisePartitions3(): #runs
    tree3 = d.buildTree(m.monk3, m.attributes)
    score3 = d.check(tree3, m.monk3test)
    print("Performance of monk3 tree: " + str(score3) + "\n")
    for index, partition in enumerate(partitions):
        for j in range(runs):
            train3, val3 = d.partition(m.monk3, partition)
            tree3a = d.buildTree(train3, m.attributes)
            best3 = bestPrunedTree(tree3a, val3)
            bigList3.append(1 - d.check(best3, m.monk3test))
        errorList3.append(sum(bigList3) / len(bigList3))
        varianceList3.append(variance(bigList3, errorList3[index]))

    return errorList3, varianceList3

# checking results
monk1mean, monk1var = optimisePartitions1()
monk3mean, monk3var = optimisePartitions3()
print(len(monk1mean))

# plotting

def plot():
    p.xlabel("Partitions")
    p.ylabel("Error on test")
    p.title('Iterations = ' + str(runs))
    for i in range(0,5):
        p.plot(partitions[i], monk1mean[i], "g^")
        p.plot(partitions[i], monk3mean[i], 'ro')
    p.show()

# producing results in plot
#plot()


def plotvar():
    p.xlabel("Partitions")
    p.ylabel("Variance on data")
    p.title('Iterations = ' + str(runs))
    for i in range(0,5):
        p.plot(partitions[i], monk1var[i], "g^")
        p.plot(partitions[i], monk3var[i], 'ro')
    p.show()

# producing results in plot
plotvar()