"""
Visualize Genetic Algorithm to find a maximum point in a function.
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 500
X_BOUND = [0,5]         # x upper and lower bounds

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function

# find non-zero fitness for selection
def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)

# 将二进制的DNA转换为十进制，并调整其大小在（0,5）之间
def translateDNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]

# 根据fitness计算出来的结果转换为概率p，根据概率p选取100个数(可重复)，对应的就是DNA的index
def select(pop, fitness):
    idx = np.random.choice(np.arange(len(pop)), size=len(pop), replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]

def crossover(parent, pop):     # mating process (genes crossover)
    i_ = np.random.randint(0, len(pop), size=1)      # 随机挑选另一个individual
    cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # 选择交叉点（True的位置交叉）
    parent[cross_points] = pop[i_, cross_points]     # 把parent中True对应位置换成individual这个位置的数值
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:       # 0.3%的概率基因突变
            child[point] = 1 if child[point] == 0 else 0
    return child

pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # 初始化DNA(二进制数)

plt.ion()       # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))    # 将DNA转换为十进制并带入F(x)

    # 画图：先清除前一个图中的散点，再画新的散点
    if 'sca' in globals():
        sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # GA part (evolution)
    # 根据给定的点算出相对最低点的位置，挑选最高点对应的DNA
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    fit_dict = {}
    for index,value in enumerate(fitness):
        fit_dict[index] = value
    # 根据fitness进行排序
    sorted_ = sorted(fit_dict.items(),key=lambda x:x[1],reverse=True)
    # 挑出fitness排名前20的DNA，下一轮迭代中保持不变
    fixed_pop = np.array([pop[i[0]] for i in sorted_[:20]])
    # 剩余的80个DNA，进行crossover和mutate
    variable_pop = np.array([pop[i[0]] for i in sorted_[20:]])
    fitness_ = np.array([fitness[i[0]] for i in sorted_[20:]])
    select_var_pop = select(variable_pop, fitness_)
    pop_copy = select_var_pop.copy()
    for parent in select_var_pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child
    # 将固定的20个pop和变换的80个pop组合成一个pop
    pop = np.concatenate((fixed_pop,select_var_pop),axis=0)

plt.ioff()
plt.show()