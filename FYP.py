import datetime as dt
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as pdr
import yfinance as yf

POPULATION_SIZE = 100
GENERATION = 100
GENES = [0, 1]
CROSSOVER_RATE = 0.1
MUTATION_RATE = 0.05
KEEP_RATE = 0.1

# end = dt.datetime.now().date()
# start = end.replace(year=end.year-1)

# msft = yf.Ticker("MSFT")
# MSFT META AMZN JNJ DUK KO     TSLA BA JPM AMT     GOOG INTC
stock = 'AMT'
start = '2021-01-01'
end = '2022-01-01'


def bits_to_values(num1, num2, num3):
    return (num1*4+num2*2+num3)*10+5


class Individual(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome
        # self.positive = 0
        self.fitness, self.trade, self.positive = self.cal_fitness()

    @classmethod
    def mutated_genes(self):
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(self):
        return [self.mutated_genes() for _ in range(11)]

    def crossover(self, par2):
        # for i in range(len(self.chromosome)):
        # prob = random.random()
        # if prob < 0.5:
        # self.chromosome[i], par2.chromosome[i] = par2.chromosome[i], self.chromosome[i]
        pos = random.randrange(len(self.chromosome))
        self.chromosome[:pos], par2.chromosome[:pos] = par2.chromosome[:pos], self.chromosome[:pos]
        self.fitness, self.trade, self.positive = self.cal_fitness()
        par2.fitness, par2.trade, self.positive = par2.cal_fitness()
        return self, par2

    def mutation(self):
        index = random.randrange(len(self.chromosome))
        self.chromosome[index] = 0 if self.chromosome[index] == 1 else 1
        self.fitness, self.trade, self.positive = self.cal_fitness()
        return self

    def cal_fitness(self):
        # RSI, KD, MACD, EMA, ROC = False
        indicator = [False, False, False, False, False]
        profit = trade = positive = 0
        for i in range(14, len(df)-3):
            if(self.chromosome[0] == 0 and df['RSI'][i] < bits_to_values(self.chromosome[1], self.chromosome[2], self.chromosome[3])):
                # fitness += df['Adj Close'][i+3] - df['Adj Close'][i]
                # RSI += 1 if df['Adj Close'][i+3] > df['Adj Close'][i] else 0
                indicator[0] = True
            if(self.chromosome[0] == 1 and df['RSI'][i] > bits_to_values(self.chromosome[1], self.chromosome[2], self.chromosome[3])):
                indicator[0] = True
            if (self.chromosome[4] == 0 and df['%K'][i] < bits_to_values(self.chromosome[5], self.chromosome[6], self.chromosome[7])):
                indicator[1] = True
            if (self.chromosome[4] == 1 and df['%K'][i] > bits_to_values(self.chromosome[5], self.chromosome[6], self.chromosome[7])):
                indicator[1] = True
            if (self.chromosome[8] == 0 and df['MACD'][i] < df['MACD_SIGNAL'][i]):
                indicator[2] = True
            if (self.chromosome[8] == 1 and df['MACD'][i] > df['MACD_SIGNAL'][i]):
                indicator[2] = True
            if (self.chromosome[9] == 0 and df['EMA5'][i] < df['EMA20'][i]):
                indicator[3] = True
            if (self.chromosome[9] == 1 and df['EMA5'][i] > df['EMA20'][i]):
                indicator[3] = True
            if (self.chromosome[10] == 0 and df['ROC'][i] < 0):
                indicator[4] = True
            if (self.chromosome[10] == 1 and df['ROC'][i] > 0):
                indicator[4] = True

            if (all(indicator)):
                trade += 1
                if df['Close'][i+3] > df['Close'][i]:
                    positive += 1
                # positive += 1 if df['Close'][i+3] > df['Close'][i] else 0
                profit += (df['Close'][i+3] - df['Close'][i]) / df['Close'][i]
                # print('profit: {}, i: {}, trade: {}'.format(
                #     (df['Close'][i+3] - df['Close'][i]) / df['Close'][i], i, trade))

            indicator = [False, False, False, False, False]

        return profit, trade, positive


#df = pdr.get_data_yahoo(stock, start=start, end=end)
df = yf.download(stock, start, end)
df['Up Move'] = np.nan
df['Down Move'] = np.nan
df['Up Move'][0] = 0
df['Down Move'][0] = 0
df['Average Up'] = np.nan
df['Average Down'] = np.nan
df['ROC'] = np.nan

# Relative Strength
df['RS'] = np.nan
# Relative Strength Index
df['RSI'] = np.nan

# Calculate Up Move & Down Move
for x in range(1, len(df)):
    df['Up Move'][x] = 0
    df['Down Move'][x] = 0

    if df['Close'][x] > df['Close'][x-1]:
        df['Up Move'][x] = df['Close'][x] - df['Close'][x-1]

    if df['Close'][x] < df['Close'][x-1]:
        df['Down Move'][x] = abs(df['Close'][x] - df['Close'][x-1])

# Calculate initial Average Up & Down, RS and RSI
df['Average Up'][13] = df['Up Move'][0:14].mean()
df['Average Down'][13] = df['Down Move'][0:14].mean()
df['RS'][13] = df['Average Up'][13] / df['Average Down'][13]
df['RSI'][13] = 100 - (100/(1+df['RS'][13]))


# Calculate rest of Average Up, Average Down, RS, RSI
for x in range(14, len(df)):
    df['Average Up'][x] = (df['Average Up'][x-1]*13+df['Up Move'][x])/14
    df['Average Down'][x] = (df['Average Down'][x-1]*13+df['Down Move'][x])/14
    df['RS'][x] = df['Average Up'][x] / df['Average Down'][x]
    df['RSI'][x] = 100 - (100 / (1+df['RS'][x]))


df['EMA5'] = df['Close'].ewm(
    span=5, min_periods=0, adjust=False, ignore_na=False).mean()
df['EMA12'] = df['Close'].ewm(
    span=12, min_periods=0, adjust=False, ignore_na=False).mean()
df['EMA20'] = df['Close'].ewm(
    span=20, min_periods=0, adjust=False, ignore_na=False).mean()
df['EMA26'] = df['Close'].ewm(
    span=26, min_periods=0, adjust=False, ignore_na=False).mean()
df['MACD'] = df['EMA12']-df['EMA26']
df['MACD_SIGNAL'] = df['MACD'].ewm(
    span=9, min_periods=0, adjust=False, ignore_na=False).mean()
df['14-high'] = df['High'].rolling(14).max()
df['14-low'] = df['Low'].rolling(14).min()
df['14-mean'] = df['Low'].rolling(14).mean()
df['%K'] = (df['Close'] - df['14-low'])*100 / (df['14-high'] - df['14-low'])
for x in range(14, len(df)):
    df['ROC'][x] = ((df['Close'][x] - df['Close'][x-14])/df['Close'][x-14])*100
# df.to_csv('C:/Users/kamwa/Desktop/msft train1.csv')
# plt.style.use('classic')
# fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))
# fig.suptitle('Stock Price (top) - %K line (bottom)')
# axs[0].plot(df['Adj Close'])
# axs[1].plot(df['RSI'])
# axs[1].plot(df['MACD'], 'b')
# axs[1].plot(df['MACD_SIGNAL'], 'r')
# axs[1].plot(df['ROC'])
# axs[1].plot(df['%K'])
# axs[1].plot(df['EMA20'], 'b')
# axs[1].plot(df['EMA5'], 'r')
# axs[0].grid()
# axs[1].grid()
# plt.show()


def main():
    population, population1, population2 = [], [], []
    for _ in range(POPULATION_SIZE):  # POPULATION_SIZE
        # gnome = Individual.create_gnome()
        population.append(Individual(Individual.create_gnome()))
        population1.append(Individual(Individual.create_gnome()))
        population2.append(Individual(Individual.create_gnome()))
        #population.append(Individual([1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]))

    # fitness from worst to best in the list
    population = sorted(population, key=lambda x: x.fitness)
    population1 = sorted(population1, key=lambda x: x.fitness)
    population2 = sorted(population2, key=lambda x: x.fitness)

    # for i in range(POPULATION_SIZE):
    #     print(i, '; ', population[i].fitness, ';',
    #           population[i].trade, ';', population[i].chromosome)
    # print("{},\tTotal: {}\n".format(
    #     population[-1].fitness, sum(i.fitness for i in population)))
    #print(sum(i.fitness for i in population))

    for i in range(GENERATION):
        new_generation, new_generation1, new_generation2 = [], [], []

        # keep certain amount of best sample
        new_generation.extend(
            deepcopy(population[int(-POPULATION_SIZE*KEEP_RATE):]))
        new_generation.extend(
            deepcopy(population[:int(POPULATION_SIZE*KEEP_RATE)]))

        new_generation1.extend(
            deepcopy(population1[int(-POPULATION_SIZE*KEEP_RATE):]))
        new_generation1.extend(
            deepcopy(population1[:int(POPULATION_SIZE*KEEP_RATE)]))

        new_generation2.extend(
            deepcopy(population2[int(-POPULATION_SIZE*KEEP_RATE):]))
        new_generation2.extend(
            deepcopy(population2[:int(POPULATION_SIZE*KEEP_RATE)]))

        # append certain amount of crossover sample with itself
        for _ in range(int(POPULATION_SIZE * CROSSOVER_RATE * 0.8)):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child1, child2 = parent1.crossover(parent2)
            new_generation.append(deepcopy(child1))
            new_generation.append(deepcopy(child2))

            parent1 = random.choice(population1)
            parent2 = random.choice(population1)
            child1, child2 = parent1.crossover(parent2)
            new_generation1.append(deepcopy(child1))
            new_generation1.append(deepcopy(child2))

            parent1 = random.choice(population2)
            parent2 = random.choice(population2)
            child1, child2 = parent1.crossover(parent2)
            new_generation2.append(deepcopy(child1))
            new_generation2.append(deepcopy(child2))

        # append certain amount of crossover sample with other populations
        for _ in range(int(POPULATION_SIZE * CROSSOVER_RATE * 0.2)):
            parent1 = random.choice(population1)
            parent2 = random.choice(population2)
            child1, child2 = parent1.crossover(parent2)
            new_generation.append(deepcopy(child1))
            new_generation.append(deepcopy(child2))

            parent1 = random.choice(population)
            parent2 = random.choice(population2)
            child1, child2 = parent1.crossover(parent2)
            new_generation1.append(deepcopy(child1))
            new_generation1.append(deepcopy(child2))

            parent1 = random.choice(population)
            parent2 = random.choice(population1)
            child1, child2 = parent1.crossover(parent2)
            new_generation2.append(deepcopy(child1))
            new_generation2.append(deepcopy(child2))

        # append certain amount of mutate sample
        for _ in range(int(POPULATION_SIZE * MUTATION_RATE)):
            new_generation.append(
                deepcopy(random.choice(population).mutation()))
            new_generation1.append(
                deepcopy(random.choice(population1).mutation()))
            new_generation2.append(
                deepcopy(random.choice(population2).mutation()))

        new_generation.extend(deepcopy(
            population[int(POPULATION_SIZE*(MUTATION_RATE+2*CROSSOVER_RATE+2*KEEP_RATE)):]))
        new_generation1.extend(deepcopy(
            population1[int(POPULATION_SIZE*(MUTATION_RATE+2*CROSSOVER_RATE+2*KEEP_RATE)):]))
        new_generation2.extend(deepcopy(
            population2[int(POPULATION_SIZE*(MUTATION_RATE+2*CROSSOVER_RATE+2*KEEP_RATE)):]))

        population = new_generation
        population1 = new_generation1
        population2 = new_generation2

        population = sorted(population, key=lambda x: x.fitness)
        population1 = sorted(population1, key=lambda x: x.fitness)
        population2 = sorted(population2, key=lambda x: x.fitness)

        # for i in range(POPULATION_SIZE):
        #     print(i, '; ', population[i].fitness, ';',
        #           population[i].trade, ';\t', population[i].chromosome)
        # print("{},Total: {}, chromosome: {}".format(
        #     population[-1].fitness, sum(i.fitness for i in population), population[-1].chromosome))
        #print(sum(i.fitness for i in population))

    result = population + population1 + population2
    result = sorted(result, key=lambda x: x.fitness)
    print(population[99].fitness, ' ', population[99].chromosome)
    print(population1[99].fitness, ' ', population1[99].chromosome)
    print(population2[99].fitness, ' ', population2[99].chromosome)
    for i in range(len(result)):
        print(result[i].fitness, '   ',result[i].trade, '   ',result[i].positive, '  ', result[i].chromosome)
    # print("Total: ", sum(i.fitness for i in population))
    # print("{}, Total: {}".format(population[-1].fitness ,sum(i.fitness for i in population) ))
    # print('Trade times {}'.format(population[-1].trade))


if __name__ == "__main__":
    main()
