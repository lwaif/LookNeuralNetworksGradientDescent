#coding:utf-8
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#
#   参数解释：
#   "pd_" ：偏导的前缀
#   "d_" ：导数的前缀
#   "w_ho" ：隐含层到输出层的权重系数索引
#   "w_ih" ：输入层到隐含层的权重系数的索引

class NeuralNetwork:
    LEARNING_RATE = 0.5
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias,self)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias,self)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)
        
        self.x_list=[]
        self.x=[]
        self.y=[]
        self.z=[]

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

	#正向传送
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. 输出神经元的值
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)#初始化数组
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. 隐含层神经元的值
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. 更新输出层权重系数
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ  w5-n*B8
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. 更新隐含层的权重系数
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error
    def show(self,x,y,z):
        cValue = ['r','y','g','b','r','y','g','b','r','y','g','b','r','y','g','b','r','y','g','b','r','y','g','b','r','y','g','b','r','y','g','b','r'] 
#        print(x,y,(x+1)*(y+1))
#        self.ax.scatter(x,y,z,c='w')
#        plt.pause(0.01)
#        cc=[]
#        for n in range(len(self.x)):
#          cc.append('r')
        self.ax.scatter(x,y,z,c=z)
    def write(self):
        self.hidden_layer.write_inspect()
        # self.output_layer.write_inspect()
    def add_show(self,x,y,z):
        b=[x,y,z]
        self.x_list.append(b)
        if len(self.x_list)%5==0:
          self.x.append(x)
          self.y.append(y)
          self.z.append(z)
    def add_show_ok(self):
        print(len(self.x))
        self.fig=plt.figure(1)#得到画面
        self.ax=self.fig.gca(projection='3d')#得到3d坐标的图
        self.show(self.x,self.y,self.z)
#        plt.show()
    def show_change(self,x,y,z,s=0):
        cValue = ['r','y','g','b',
        'r','y','g','b',
        'r','y','g','b',
        'r','y','g','b',
        'r','y','g','b',
        'r','y','g','b',
        'r','y','g','b',
        'r','y','g','b','r'] 
#        print(x,y,(x+1)*(y+1))
#        self.ax.scatter(x,y,z,c='w')
#        plt.pause(0.01)
        if s==0:
            self.ax.scatter(x,y,z,c=cValue[(x+1)])     
        else:
            self.ax.scatter(x,y,z,c=cValue[(y+1)])     
#        self.ax.scatter(x,y,z,c=cValue[(x+1)*(y+1)])     
    def add_show_change(self,s=0):
        print(len(self.x))
        self.fig=plt.figure(2)#得到画面
        self.ax=self.fig.gca(projection='3d')#得到3d坐标的图
        t=1
        for n in range(len(self.x_list)):
#         if n % 8 == 0:
#          if n % 10 == 0:
            if s==0:
                self.show_change(self.x_list[n][0],self.x_list[n][1]+t,self.x_list[n][2])
            else:
                self.show_change(self.x_list[n][0]+t,self.x_list[n][1],self.x_list[n][2],s)
            t+=1
            plt.pause(0.000001)
            print(n, self.x_list[n][0],self.x_list[n][1]+t,self.x_list[n][2])
        while True:
          plt.pause(0.05) 
            
class NeuronLayer:
    def __init__(self, num_neurons, bias,p):
        self.parent=p
        
        # 同一层的神经元共享一个截距项b
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
    def write_inspect(self):
        for n in range(len(self.neurons)):
#        n=0
            for w in range(len(self.neurons[n].weights)):
                self.parent.add_show(n,w, self.neurons[n].weights[w])
                
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:',n,w, self.neurons[n].weights[w])
#                self.parent.show(n,w, self.neurons[n].weights[w])

            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
#        print(self.output)
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # 激活函数sigmoid
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

	#计算误差总输入
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # 每一个神经元的误差是由平方差公式计算的
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    #计算输出误差
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    #计算总网络输入
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)


    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]
def f_study(in_data,out_data,num_hidden):
    # hidden_layer_weights=[0.5]*(num_hidden*len(in_data))
    # output_layer_weights=[0.5]*(num_hidden*len(out_data))
    hidden_layer_weights=[random.random()]*(num_hidden*len(in_data))
    output_layer_weights=[random.random()]*(num_hidden*len(out_data))
    for i in range(len(hidden_layer_weights)):
        # hidden_layer_weights[i]=random.random()
        hidden_layer_weights[i]=random.random()
    for i in range(len(output_layer_weights)):
        output_layer_weights[i]=random.random()
        # output_layer_weights[i]=0.5
    hidden_layer_bias=random.random()
    output_layer_bias=random.random()
    # hidden_layer_bias=0.5
    # output_layer_bias=0.5
    #hidden_layer_bias=0.9818422593240586
    #output_layer_bias=0.4257418298889508
    #output_layer_weights=[0.21060908544579016, 0.3621015782256518, 0.389550608461516, 0.694887632535093, 0.2838883570278381, 0.14531159931211668, 0.050878199748789266, 0.9347604317649699]
    #hidden_layer_weights=[0.04280793666627325, 0.08852978656172383, 0.37410852669722616, 0.971427209397932, 0.9329954834047123, 0.19868833549666387, 0.2074656195413116, 0.9559695380687594]    
    nn=NeuralNetwork(len(in_data),num_hidden,len(out_data),hidden_layer_weights,hidden_layer_bias,output_layer_weights,output_layer_bias)
    for i in range(1000):
    # for i in range(random.randint(5000,10000)):
        nn.train(in_data, out_data)
        data = nn.calculate_total_error([[in_data, out_data]])
        wc = round(data, 9)
        if i % 10 == 0:
            nn.write()
        if i % 100 == 0:
            print(i, round(nn.calculate_total_error([[in_data, out_data]]), 9))
    print(i,nn.feed_forward(in_data))
    nn.inspect()
    nn.add_show_ok()
    nn.add_show_change(s=1)

    return nn

if 1:
 num_hidden=8#random.randint(2,15)
 print(num_hidden)
# aa = f_study([0.1,0.2], [0.4, 0.3],num_hidden)    
 aa = f_study([0.1,0.2,0.3,0.4, 0.5, 0.7], [0.74, 0.8,0.54,0.01],num_hidden)    
