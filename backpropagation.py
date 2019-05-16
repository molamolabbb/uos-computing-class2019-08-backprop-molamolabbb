from math import exp
from copy import deepcopy
from random import random
from randomness import randnr
import csv
rand = randnr(3)
def in_random_order(x):
    #"Returns an iterator that presents the list x in a random order"
    indices = [i for i, _ in enumerate(x)]
    # "inside-out" Fisher-Yates shuffle. Step through the list, and at
    # each point, exchange the current element with a random element
    # in the list (including itself)
    for i in range(len(indices)):
        j = (rand.randint() // 65536) % (i+1)  # The lower bits of our random generator are correlated!
        indices[i], indices[j] = indices[j], indices[i]
    for i in indices:
        yield x[i]

def sigmoid(x):
    try:
        return 1.0 / (1.0 + exp(-x))
    except OverflowError:
        print("OV", x)
        raise OverflowError

def multi_best(xs):
    best = max(xs)
    return [1 if x == best else 0 for x in xs]

def multi_accuracy(data, target, f):
    tot = 0
    cor = 0
    for d, t in zip(data, target):
        if multi_best(f(d)) == t:
            cor += 1
        tot += 1
    return cor / float(tot)

def inner_neuron(weight, input_):
    return weight[0]+sum(weight[i+1]*input_[i] for i in range(len(input_)))

def sigmoid_neuron(weight, input_):
    return sigmoid(inner_neuron(weight, input_))

def initialize_weights(n_nodes, initialize_fn = lambda : random()):
    ini_fn = initialize_fn
    a = [ [] for i in range(len(n_nodes)-1)]
    for n in range(len(n_nodes)-1):
        a[n] = [[] for i in range(n_nodes[n+1])]
        for j in range(n_nodes[n+1]):
            a[n][j] = []
            for i in range(n_nodes[n]+1):
                a[n][j].append(initialize_fn())
    return a 

def feedforward_(network, inputs, hidden_neuron=sigmoid_neuron, output_neuron=sigmoid_neuron):
    result = []
    input_vector = inputs
    result.append(input_vector)
    if len(network)>1:
        hidden_layer = network[:-1]
        for i in range(len(hidden_layer)):
            hidden_result = [hidden_neuron(hidden_layer[i][j], input_vector) for j in range(len(hidden_layer[i]))]
            input_vector = hidden_result
            result.append(hidden_result)
    output_layer = network[-1]
    out_result = []
    for output in output_layer:
        out_result.append(output_neuron(output,input_vector))
    result.append(out_result)
    return result

def feedforward(network, inputs):
    return feedforward_(network, inputs)[-1]

def calculate_deltas(network, activations, y): 
    n_nodes = [len(activations[i]) for i in range(len(activations))]
    delta = [[] for i in range(len(n_nodes)-1)]
    for l in range(len(n_nodes)-1,0,-1):
        for j in range(n_nodes[l]):
            if l ==  len(n_nodes)-1:
                delta[l-1].append((activations[l][j]-y[j])*activations[l][j]*(1-activations[l][j]))                
            else :
                delta[l-1].append(activations[l][j]*(1-activations[l][j])*(sum(network[l][k][j]*delta[l][k] for k in range(n_nodes[l+1]))))
    return delta

def batch_update_nn(network_, activations, deltas, eta):
    network = deepcopy(network_)
    n_nodes = [len(activations[i]) for i in range(len(activations))]
    #print (n_nodes)
    for l in range(len(n_nodes)-1):
        for k in range(n_nodes[l+1]):
            network[l][k][0] -= eta*deltas[l][k]
            for j in range(n_nodes[l]-1):
                #print(n_nodes[l])
                #print (j)
                network[l][k][j+1] -= eta*activations[l][j]*deltas[l][k]
    return network

# You can use the structure from before (reduce alpha0 every time), or
# you can just run over the data n_epochs times, and then choose a new
# alpha by hand
def sgd_nn(x, y, theta_0, alpha_0: float = 0.01, iterations=20):
    data = list(zip(x, y))
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0
    iter = 0
    for i in range( iterations):
        #value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
        value = 0
        for x_i, y_i in data :
            feedfo = feedforward(theta, x_i)
            value += sum((y_i[j]-feedfo[j])**2 for j in range(len(y_i)))/2
        if iter % 100 == 0: print(iter, value, theta)
        iter += 1
        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9
            theta = min_theta
        for x_i, y_i in in_random_order(data):
            feed_ = feedforward_(theta, x_i)
            #gradient_i = list(gradient_fn(x_i, y_i, theta))
            gradient_i = calculate_deltas(theta, feed_, y_i)
            #theta = list(t - alpha*g for t, g in zip(theta, gradient_i))
            theta = batch_update_nn(theta, feed_, gradient_i, eta=0.1)
    return min_theta,0

if __name__ == "__main__":
    # Setup for Fisher's iris classification task
    iris = csv.reader(open('Fisher.txt'), delimiter='\t')
    header = iris.__next__()  # change to iris.next() for python2!
    data_ = list(d for d in iris)
    data = list([[float(di) for di in d[1:]] for d in data_])
    target = [[0,0,0] for _ in data_]
    for i, di in enumerate(data_):
        target[i][int(di[0])] = 1
    # Example to setup and use a neural network for Fisher.
    # Further examples in test_backpropagation
    # Network architecture: 4 > 8 > 3
    network_ = initialize_weights([4, 8, 3], lambda: rand.gauss(0, 1))
    #network_ = [[[-0.6982309887519852, -0.5639800862554718, -1.2319230681874833, 1.396812550787976, -0.039664383152258335], [-1.3627717782569528, 0.011778612450230128, -0.5425878499055428, 1.3566263275302217, -1.3890966920829155], [0.7094613663721133, -0.8441274555764758, 2.7720787209982434, 3.9183244167605236, 2.467619442751003], [0.41660343375151015, 0.6187470858442943, -0.11528513294554307, -2.07493324998818, -0.6547767790020792], [0.03994498340357939, 0.9910363908137124, -1.7728343342373596, 0.5385407866381259, -0.20400763400016142], [-0.7833733467238596, -2.911803337004647, -0.5539255272532548, 0.3378397915028057, -0.7849008431177941], [-1.6066758442326932, -0.21804947753121015, 1.0473981896329163, 1.1346036133381465, 0.3995418938003271], [-0.6899276351369494, 0.6065715191608547, 0.7375160148264339, -1.5345267576895878, -1.1309646407038743]], [[-1.3603868875301577, 9.130272642582772, -1.3093176066666818, -1.374550972934381, -0.2959882073088689, -0.19769074678234225, -0.40854428884305877, -1.9671425642194096, 0.6034815536141308], [-0.08075200422708662, -4.778115274760305, -0.7684973054923281, -0.3121989794712119, -0.48707923434141986, 2.421793162987332, 0.7180772858671228, 0.3924000558564009, -1.4501180631298003], [0.21112852597624474, -4.77905571716167, 0.6570467765430429, 0.401609897128214, 0.1017823284401708, 0.214760388745296, 0.5416358715340643, -0.6132881541029576, -0.8856737362237809]]]
    network, l = sgd_nn(data, target, network_, 0.01, 10000)
    # Test out, show sgd has improved
    print("Before training:")
    print(multi_accuracy(data, target, lambda x: feedforward(network_, x)))
    print("After training:")
    print(multi_accuracy(data, target, lambda x: feedforward(network, x)))
    fout = open('results.txt', 'w')
    fout.write('accuracy : %f\n' % (multi_accuracy(data, target, lambda x: feedforward(network, x))))
    fout.write('best network : {0}'.format(network))
    fout.close()


