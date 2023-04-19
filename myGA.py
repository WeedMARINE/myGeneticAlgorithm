#Napongkorn Suvanphatep
#last Edit 4/8/2023 : attempt ad adding non continuous, sample from a list design variable


import numpy as np
from pathos.multiprocessing import ProcessPool
from pathos.helpers import cpu_count
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Design_string:
    continuous_dv_list: list[float]
    custom_dv_list: list

    def tolist(self):
        return self.continuous_dv_list + self.custom_dv_list
    
    def show(self):
        print(self.continuous_dv_list,self.custom_dv_list)

class myGA:
    def __init__(self, maxGen: int,numString: int, numParent: int, numOffspring: int, TOL: float, num_continuous_dv: int, varMin, varMax, procFunction: callable, objFunction: callable, num_custom_dv:int  =0, custom_dv_gen_Function:callable = None, custom_dv_breed_Function:callable = None,  parallel_simulation:bool = False, parallel_cost:bool = False, node:int = -1) -> None:
        ##Input Description##
        #maxGen : integer, maximum total generations (terminates code if TOL is not reached)
        #numString : integer, the number of total design strings per generation
        #numParent : integer, the number of design strings to preserve and breed, divisible by 2 less than S
        #numOffspring : integer, the number of offspring
        #TOL : float, the acceptable cost function threshold to stop evolution
        #num_continuous_dv : integer, the number of design variables per string
        #procfunction : function, process function
        #objFunction : function, objective function Π(x)
        #varMin : float, minimum value of design variable, default -20.0
        #varMax : float, Maximum value of design variable, default 20.0
        self._maxGen = maxGen
        self._numString = numString
        self._numParent = numParent
        self._numOffspring = numOffspring
        self._TOL = TOL
        self._num_continuous_dv = num_continuous_dv
        self._varMin = varMin
        self._varMax = varMax
        self._procFunction = procFunction
        self._objFunction = objFunction

        self._silent_flag = False
        
        self._parallel_sim_flag = False
        if parallel_simulation == True:
            self._parallel_sim_flag = True
        
        self._parallel_cost_flag = False
        if parallel_cost == True:
            self._parallel_cost_flag = True

        self._num_custom_dv = num_custom_dv
        if self._num_custom_dv > 0:
            self._custom_dv_gen = custom_dv_gen_Function
            self._custom_dv_breed = custom_dv_breed_Function

        if self._parallel_cost_flag or self._parallel_sim_flag:
            if node < 1:
                self._node = cpu_count() - 5
            else:
                self._node = node

        self.__reset()
        pass

    def __reset(self):
    #This function reset the GA instance, It is called every time the run GA is run#
        self._break_flag = False
        
        #Array for storing lastest gen genetic strings
        # self.Lambda = np.zeros((self._numString,self._num_continuous_dv))
        self.Lambda = [Design_string([],[]) for  i in range(self._numString)]

        #Array here Pi(g, s) is the cost of the s’th-ranked design in the g’th generation.
        self.Pi = np.zeros((self._maxGen,self._numString))

        #Array where Orig(g, :) represents the indices of each sorted entry from before they were sorted
        self.Orig = np.zeros((self._maxGen,self._numString), dtype = int)
        self.Orig[0,:] = np.arange(self._numString)
        
        # #Array for storing lastest gen objective function value
        self._new_pi = np.zeros((self._numString))

        #Pi minimum, average of parent , and average of the population in each Generation
        self._pi_min = np.zeros(self._maxGen)
        self._pi_parent_avg = np.zeros(self._maxGen)
        self._pi_avg = np.zeros(self._maxGen)
        
        #generation counter
        self._genCount = 0
        
    def __arrayRand(self, n: int):
    #This function create n by num_continuous_dv(number of numerical variables in the string) array accoding to the max/min value specified during instance initialization.
        continuous_dv_arr = np.zeros((n,self._num_continuous_dv))
        custom_dv_arr = []
        for i in range(n):
            #generate continuous design variables
            for j in range(self._num_continuous_dv):
                if self._num_continuous_dv > 1:
                    continuous_dv_arr[i,j] = random.uniform(self._varMin[j],self._varMax[j])
                else:
                    continuous_dv_arr[i,j] = random.uniform(self._varMin,self._varMax)
            #generate custom design variables
            if self._num_custom_dv > 0:
                if self._num_custom_dv == 1:
                    custom_dv_arr.append([self._custom_dv_gen()])
                else:
                    custom_dv_arr.append(list(self._custom_dv_gen()))
        continuous_dv_arr = continuous_dv_arr.tolist()
        if self._num_custom_dv > 0:
            new_design_strings = [Design_string(continuous_dv_arr[i],custom_dv_arr[i]) for  i in range(n)]
        else:
            new_design_strings = [Design_string(continuous_dv_arr[i],[]) for  i in range(n)]
        return new_design_strings

    def __simulate_process(self,design_strings_list):
    # This function called the specified process function to be simulate before calculating cost. The option to do this process parallely is specify dduring GA instance initailization.
    # Running simulation in parallel resulted in 2-3 time faster run time
        input_list = list(zip(*[string.tolist() for string in design_strings_list])) #cast design_string to list and transpose the list in to vectorized argument format

        if self. _parallel_sim_flag == False:
            results = list(map(self._procFunction, *input_list))
        else:
            pool = ProcessPool(nodes=self._node)
            results = pool.map(self._procFunction, *input_list)
        return results
    
    def __calculate_cost(self,results):
    # This function called the specified cost function for calculating . The option to do this process parallely is specify dduring GA instance initailization.
    #Generally, you dont need to run it in parallel since cost function is simple and can be easily vectorized
        input_list = list(zip(*results)) #transpose the result list in to vectorized argument format
        
        if self. _parallel_cost_flag == False:
            costs = list(map(self._objFunction, *input_list))
        else:
            pool = ProcessPool(nodes=self._node)
            costs = pool.map(self._objFunction, *input_list)
        return np.asarray(costs)
    
    #for Debug
    # def arrRand(self, n:int):
    #     return self.__arrayRand(n)
    # def simulate_process(self,design_string):
    #     return self.__simulate_process(design_string)
    # def calculate_cost(self,results):
    #     return self.__calculate_cost(results)
    
    def __genrationSort(self):
    # This function sort the cost value from minimum to maximum, the sort the design string based on sorted costs     
        # the indices of each sorted Pi from before they were sorted
        sorted_Pi_indx = self._new_pi.argsort()
        #sorted _new_pi and Lambda in descending order according to Evaluated Pi Value
        #TODO allow Maximixing Sorting as specify at the instance's initalization
        sorted_new_pi = self._new_pi[sorted_Pi_indx[::]]
        # sorted_Lambda = self.Lambda[sorted_Pi_indx[::]]
        sorted_Lambda = [self.Lambda[i] for i in sorted_Pi_indx[::]]
        #Update Pi, Orig, Lambda, _pi_min, and _pi_avg Array
        self.Pi[self._genCount, :] = sorted_new_pi
        self.Orig[self._genCount, :] = sorted_Pi_indx
        self.Lambda = sorted_Lambda 
        self._pi_min[self._genCount] = np.amin(sorted_new_pi)
        self._pi_avg[self._genCount] = np.mean(sorted_new_pi)
        self._pi_parent_avg[self._genCount] = np.mean(sorted_new_pi[0:self._numParent])

    def __createNextGen(self):
    # This function generate design strings to be introduced in the next generation, which comprise of offsprings and randomly gnerated strings.
    # It also simulate the process, calculate the cost, sort the newly introduce string, and update the generation population (lambda and Pi)
        #preserved parents
        preservedParent = self.Lambda[:self._numParent]
        preservedParent_pi = self.Pi[self._genCount - 1, :self._numParent]

        parentpairs = int(self._numParent/2)
        parent_continuous_dv = np.asarray([preservedParent[i].continuous_dv_list for i in range(self._numParent)])
        offspring_continuous_dv_arr = np.zeros((self._numOffspring,self._num_continuous_dv))
        offspring_custom_dv_arr = []
        
        for pair in range(parentpairs):
            psi = np.random.rand(2)
            for var in range(self._num_continuous_dv):
                offspring_continuous_dv_arr[pair*2,var] = psi[0]*parent_continuous_dv[pair*2,var] + (1-psi[0])*parent_continuous_dv[pair*2+1,var]
                offspring_continuous_dv_arr[pair*2+1,var] = psi[1]*parent_continuous_dv[pair*2,var] + (1-psi[1])*parent_continuous_dv[pair*2+1,var]

            if self._num_custom_dv > 0:    
                kids_custom_dv = self._custom_dv_breed(preservedParent[pair].custom_dv_list,preservedParent[pair+1].custom_dv_list)
                for kid_custom_dv in kids_custom_dv:
                    offspring_custom_dv_arr.append(kid_custom_dv)

        offspring_continuous_dv_arr = offspring_continuous_dv_arr.tolist()
        if self._num_custom_dv > 0:
            offspring = [Design_string(offspring_continuous_dv_arr[i],offspring_custom_dv_arr[i]) for  i in range(self._numOffspring)]
        else:
            offspring = [Design_string(offspring_continuous_dv_arr[i],[]) for  i in range(self._numOffspring)]

        #simulate process with offspring's genetic strings
        offspring_results = self.__simulate_process(offspring)
        
        #Calculate Offspring's PI
        # offspring_pi = self._objFunction(*offspring.T).flatten()
        offspring_pi = self.__calculate_cost(offspring_results)

        #populate remaining generation population with random strings
        randomPop =  self.__arrayRand(self._numString-2*self._numParent)
        
        #simulate process with randompop's genetic strings
        randomPop_results = self.__simulate_process(randomPop)
        
        #Calculate randompop's PI
        # randomPop_pi = self._objFunction(*randomPop.T).flatten()
        randomPop_pi = self.__calculate_cost(randomPop_results)



        #update Lambda and _new_pi
        self.Lambda = preservedParent + offspring + randomPop
        self._new_pi = np.concatenate((preservedParent_pi,offspring_pi,randomPop_pi))

    def __consoleReporting(self):
    #This function is called every generation to inform the user of the progress.
        if self._silent_flag:
            pass
        else:
            # Reporting Results in the terminal
            print('Generation no: ' + str(self._genCount))
            print('This generation pi_min = ' + str(self._pi_min[self._genCount]))
            print('This generation pi_avg = ' + str(self._pi_avg[self._genCount]))
            if self._break_flag:
                print('TOL achieved at generation ' + str(self._genCount)+'!')
            else:
                if (self._genCount+1) == self._maxGen:
                    print('Did not achieved TOL')

    def plot_Pi(self):
    #This function plot minumum cost, parent average, and average at each generation.
        fig = plt.figure()
        h_axe = np.arange(0,self._genCount+1)
        ax = fig.add_subplot()
        plt.plot(h_axe,self._pi_min[:self._genCount+1], 'r',label = 'Pi_min')
        plt.plot(h_axe,self._pi_avg[:self._genCount+1], 'g',label = 'Pi_avg')
        plt.plot(h_axe,self._pi_parent_avg[:self._genCount+1], 'b',label = 'Pi_parent_avg')
        # for k in k_list:
        #     index = k+1
        #     hist = np.array(histA[index])
        #     iter = np.arange(1,len(histA[index])+1)
        #     lab = str(k)
        #     # plt.plot(iter,histA[index], marker = 'x',label = 'k = ' +lab)
        #     plt.plot(histA[index],obj_A(hist),'x--',label = 'k = ' +lab)
        ax.set_yscale('log')
        plt.legend()
        plt.xlabel('Generation [i]')
        plt.ylabel('Pi(x_i)')
        plt.title("Cost of the best design and the mean cost of all of the design strings for each generation")
        fig.show()

    def run(self,silent:bool = False):
    #Call this function to run GA optimization
        if silent:
            self._silent_flag = True
        
        self.__reset()

        #Randomized First Generation
        #Randomize value for first generation strings
        self.Lambda = self.__arrayRand(self._numString)
        
        #Simulate the process with each genetic string
        results = self.__simulate_process(self.Lambda)

        #Evaluate objective function for each genetic string
        # self._new_pi = self._objFunction(*self.Lambda.T).flatten()
        self._new_pi = self.__calculate_cost(results)
        
        #sort and Update Outputs
        self.__genrationSort()
        self.__consoleReporting()
        
        #loop for 2nd - n generations
        while (self._genCount+1) < self._maxGen:
            self._genCount+=1 #increase generation counter
            self.__createNextGen() #create new generation
            self.__genrationSort() #Elaluate ageneration and update Pi, Orig, Lambda, _pi_min, and _pi_avg Array
            if self._pi_min[self._genCount] < self._TOL:
                self._break_flag = True
                #Filling the remaining generation with the best generation's data
                self.Pi[self._genCount+1:,:] = self.Pi[self._genCount,:]
                self.Orig[self._genCount+1:,:] = self.Orig[self._genCount,:]
                self._pi_avg[self._genCount+1:] = self._pi_avg[self._genCount]
                self._pi_min[self._genCount+1:] = self._pi_min[self._genCount]
                self.__consoleReporting()
                break
            else:
                self.__consoleReporting()

