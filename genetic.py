import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import math


class Graph:
    def __init__(self,v):
        self.v = v
        self.e = 0
        self.graph = defaultdict(list)

    def addEdge(self, s, d):
        self.graph[s].append(d)
        self.e = self.e + 1

    def isReachable(self,src,dest):
        visited =[False]*(self.v+1)

        queue=[]                       
        queue.append(src)              

        visited[src] = True
        while queue:
            n = queue.pop(0)            
            if n == dest:
                   return True
            for i in self.graph[n]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        return False
    
    def printGraph(self):
        for i in range(self.v):
            print("src=",i)
            for j in self.graph[i]:
                print(j)

    def DFSUtil(self, v, visited):
        visited.add(v)
        print(v, end=' ')
 
        for j in self.graph[v]:
            if j not in visited:
                self.DFSUtil(j, visited)
 
    def DFS(self, v):
        visited = set()
        self.DFSUtil(v, visited)


# GET INPUT FUNCTION and COVERT TO GRAPH
file_address = "C:/Users/Ashkan/Desktop/Term 8/هوش/تمرین ها/تمرین 2 - بخش اول/SA/input.txt"

lines = []
with open(file_address) as f:
    lines = f.readlines() 

V = int(lines[0])
graph = Graph(V+1)
count = 0
for line in lines:
    if(count!=0):
        edge = line.split(' ')
        graph.addEdge(int(edge[0]),int(edge[1]))
    count = count + 1

def objective(sol):
    q = 0
    for i in range(n): 
        for j in range(i,n):
            if(graph.isReachable(sol[j],sol[i]) and i!=j):
                q = q + 1
    return graph.e-q
    
def cross_over(parent1, parent2):
    temp1 = np.random.randint(n)
    temp2 = np.random.randint(n)
    while (temp1==temp2):
        temp1 = np.random.randint(n)
        temp2 = np.random.randint(n)
    if temp2 < temp1:
        temp1,temp2 = temp2,temp1

    # CREATE CHILD 1 
    child1 = np.zeros(n,dtype=int)
    removed = child1.copy()
    for i in range(temp1, temp2+1):
        child1[i] = parent1[i]
        removed[parent1[i]-1] = 1

    i = 0
    j = 0
    while i < n and j < n:
        if i >= temp1 and i <= temp2:
            i = i + 1
            continue
        while j < n and removed[parent2[j]-1]:
            j = j + 1
        if j == n:
            break

        child1[i] = parent2[j]
        i = i + 1
        j = j + 1


    #CREATE CHILD 2
    child2 = np.zeros(n,dtype=int)
    removed = child2.copy()
    for i in range(temp1, temp2 + 1): 
        child2[i] = parent2[i]
        removed[parent2[i]-1] = 1

    i = 0
    j = 0
    while i < n and j < n:
        if i >= temp1 and i <= temp2:
            i = i + 1
            continue
        while j < n and removed[parent1[j]-1]:
            j = j+ 1
        if j == n:
            break

        child2[i] = parent1[j]
        i = i + 1
        j = j + 1

    return (child1, child2)        
        



# INITIAL
n = V
population_size = 200
max_pop_size = 600
crossover_coeff = 0.7
mutation_coeff = 0.04
max_iteration = 500

num_crossover = round(population_size * crossover_coeff)
num_mutation = round(population_size * mutation_coeff)
total = population_size + num_crossover + num_mutation

population = []
object_values = []
best_objectives = 0
best_chromosome = np.zeros(n)

# INITAIL K Population
while len(population) < population_size:
    sequence = [i for i in range(1,n+1)]
    solution = random.sample(sequence, n)
    population.append(solution.copy())
    object_values.append(objective(solution))

# MAIN LOOP
iteration = 0
best_objective_plot = []
while iteration < max_iteration:
    summation = sum(object_values)
    pr = []
    cumulative_pr = []
    for i in range(population_size):
        pr.append(object_values[i] / summation)
    cumulative_pr.append(pr[0])
    for i in range(1, population_size - 1):
        temp = cumulative_pr[i - 1] + pr[i]
        cumulative_pr.append(temp)
    cumulative_pr.append(1)

    # CROSS OVER
    for i in range(0, num_crossover, 2):
        p1 = 0
        temp = np.random.rand()
        while(np.random.rand()==0):
            temp = np.random.rand()
        while cumulative_pr[p1] < temp:
            p1 = p1 + 1
        p2 = p1
        while p1 == p2:
            temp = np.random.rand()
            while(np.random.rand()==0):
                temp = np.random.rand()
            p = 0
            while cumulative_pr[p] < temp:
                p = p + 1
            p2 = p
        parent1 = population[p1]
        parent2 = population[p2]

        children = cross_over(parent1, parent2)
        
        child1 = children[0]
        child2 = children[1]

        population.append(child1)
        object_values.append(objective(child1))

        population.append(child2)
        object_values.append(objective(child2))

    # MUTATION
    for i in range(num_mutation):
        temp = np.random.randint(num_crossover)
        temp = temp + population_size
        mutated = population[temp]

        # MUTATION -> shuffle 2 places 
        temp = np.random.randint(n)
        temp2 = np.random.randint(n)
        while ((graph.isReachable(mutated[temp],mutated[temp2]) and temp<temp2)):
            temp = np.random.randint(n)
            temp2 = np.random.randint(n)

        mutated[temp] , mutated[temp2] = mutated[temp2] , mutated[temp]

        population.append(mutated)
        object_values.append(objective(mutated))

    best_objective = max(object_values)
    best_arg = np.argmax(object_values)
    best_chromosome = population[best_arg]

    if len(population) > max_pop_size:
        temp_population = []
        temp_objective = []
        args = np.argsort(object_values)
        for i in range(max_pop_size):
            temp = len(population) - 1 - i
            temp_population.append(population[args[temp]])
            temp_objective.append(object_values[args[temp]])
            
        population = temp_population
        object_values = temp_objective
        population_size = max_pop_size


    #print(best_objective)
    best_objective_plot.append(best_objective)
    if (best_objective == graph.e):
        break
    
    iteration = iteration + 1

print(best_chromosome)
print(best_objective)

plt.plot(best_objective_plot)
plt.show()



















