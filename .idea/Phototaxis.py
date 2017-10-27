import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import time
import pylab
import numpy as np
from threading import Condition
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pyNN.random import RandomDistribution as rand
import spynnaker8.spynakker_plotting as splot
import csv
import pandas

#run setup
seed_population = False
copy_population = False
only_improve = False
total_runtime = 2000
time_slice = 100
pop_size = 15
agent_neurons = 6
neuron_pop_size = 1
ex_in_ratio = 4#:1
visual_discrete = 2
visual_field = (2./6.)*np.pi
max_poisson = 300
mutation_rate = 0.02
shift_ratio = 0.2
number_of_children = 75
fitness_offset = 150
#maybe gentically code this
visual_weight = 4
visual_delay = 1

#params per neurons - number of necessary genetic bits

#weight per connection - n*n
weights = agent_neurons * agent_neurons
weight_min = 0
weight_max = 5
weight_range = weight_max - weight_min
weight_cut = 0
#delays per neruon connection - n*n
delays = agent_neurons * agent_neurons
delay_loc = weights
delay_min = 1
delay_max = 144
delay_range = delay_max - delay_min
#inhibitory on off
inhibitory = 1
#weights set-able to 0
set2zero = agent_neurons * agent_neurons
connects_p_neuron = 4.0
set2chance = connects_p_neuron/agent_neurons
#plasticity on off - 1
plasticity = 1
plastic_prob = 1
#plasticity per neuron - n*n
plasticity_per_n = 0
#net size? - 1
net_size = 0
#cell params? - 1 (n*n)
cell_params = 0
#recurrancy? - 1 (n*n)
recurrency = 1
#environment data (x, y, theta), for now static start
status = 3
x_centre = 0
x_range = 0
y_centre = 0
y_range = 0
angle = 0
angle_range = 0
#light configuration
light_dist_min = 50
light_dist_range = 200
light_angle = 0 #to 2*pi
random_light_location = True

print "can I commit"

port_offset = 1
number_of_runs = 2
counter = 1
child = pop_size
neuron_labels = list()

genetic_length = weights + delays + (inhibitory * agent_neurons) + set2zero + \
                 plasticity + plasticity_per_n + net_size + cell_params + status

#intialise population
agent_pop = [[0 for i in range(genetic_length)] for j in range(pop_size+1)] #[pop][gen]
temp_pop = [[0 for i in range(genetic_length)] for j in range(pop_size + 1)]
if seed_population == True:
    if inhibitory != 0:
        inhibitory_loc = weights + delays
        set2loc = inhibitory_loc + agent_neurons
    else:
        set2loc = weights + delays
    if copy_population == True:
        pop_fitness = [0 for i in range(pop_size)]
        print "copying whole population"
        with open('Seed population.csv') as from_file:
            csvFile = csv.reader(from_file)
            i = 0
            for row in csvFile:
                temp = row
                pop_fitness[i] = temp[0]
                for j in range(genetic_length):
                    agent_pop[i][j] = float(temp[j+1])
                i += 1
    else:
        print "seeding from the best individual"
        with open('Seed population.csv') as from_file:
            csvFile = csv.reader(from_file)
            for row in csvFile:
                temp = row
                for j in range(genetic_length):
                    agent_pop[0][j] = float(temp[j+1])
                break
            for i in range(pop_size-1):
                print "mutating individuals"
                for j in range(genetic_length):
                    agent_pop[i+1][j] = agent_pop[0][j]
                    if j < weights:
                        if recurrency == 0 and i % (agent_neurons + 1) == 0:
                            agent_pop[i+1][j] = 0;
                        else:
                            agent_pop[i+1][j] = np.random.normal(agent_pop[i+1][j], weight_range * shift_ratio)
                            if agent_pop[i+1][j] < weight_min:
                                agent_pop[i+1][j] += weight_range
                            if agent_pop[i+1][j] > weight_max:
                                agent_pop[i+1][j] -= weight_range
                            if agent_pop[i+1][j] < weight_cut:
                                agent_pop[i+1][j] = 0
                    elif j < weights + delays:
                        if recurrency == 0 and (i - 1) % (agent_neurons + 1):
                            agent_pop[i+1][j] = 0;
                        else:
                            agent_pop[i+1][j] = np.random.normal(agent_pop[i+1][j], delay_range * shift_ratio)
                        if agent_pop[i+1][j] < delay_min:
                            agent_pop[i+1][j] += delay_range
                        if agent_pop[i+1][j] > delay_max:
                            agent_pop[i+1][j] -= delay_range
                    elif j < inhibitory_loc + agent_neurons:
                        if np.random.normal(0, 1) < (1 / float(ex_in_ratio + 1)):
                            agent_pop[i+1][j] *= -1
                    elif j < set2loc + set2zero:
                        if np.random.uniform(0, 1) > set2chance: #only works well if set2chance > 0.5
                            if agent_pop[i+1][j] == 1:
                                agent_pop[i+1][j] = 0
                            else:
                                agent_pop[i+1][j] = 1
                    elif j < set2loc + set2zero + 1:
                        if np.random.uniform(0, 1) < plastic_prob:
                            agent_pop[i+1][j] = 0
                        else:
                            agent_pop[i+1][j] = 1
                    elif j < set2loc + set2zero + 4:
                        print "setting location variables"
                    else:
                        print "shouldn't be here, location saved for further genetic manipulation"
else:
    for i in range(pop_size):
        j = 0
        #initialise weights
        while j < weights:
            if recurrency == 0 and j%(agent_neurons+1):
                agent_pop[i][j] = 0;
            else:
                agent_pop[i][j] = np.random.uniform(weight_min, weight_max)
                if agent_pop[i][j] < weight_cut:
                    agent_pop[i][j] = 0
            j += 1
        #initilaise delays
        while j < weights + delays:
            if recurrency == 0 and (j-1)%(agent_neurons+1):
                agent_pop[i][j] = 0;
            else:
                agent_pop[i][j] = np.random.uniform(delay_min, delay_max)
            j += 1
        #set inhibitory neurons
        if inhibitory !=0:
            inhibitory_loc = weights + delays
            while j < inhibitory_loc + agent_neurons:
                if np.random.uniform(0, 1) < (1 / float(ex_in_ratio + 1)):
                    agent_pop[i][j] = -1
                else:
                    agent_pop[i][j] = 1
                j += 1
            set2loc = inhibitory_loc + agent_neurons
        else:
            set2loc = weights + delays
        #enable connection between neurons or not
        while j < set2loc + set2zero:
            if np.random.uniform(0,1) < set2chance:
                agent_pop[i][j] = 1
            else:
                agent_pop[i][j] = 0
            j += 1
        #plasticity on off
        if np.random.uniform(0, 1) < plastic_prob:
            agent_pop[i][j] = 0
        else:
            agent_pop[i][j] = 1
        j += 1
        #initialise plastic value per neuron
        if plasticity_per_n != 0:
            #set some values ina  while loop
            nothing = 0
        if net_size != 0:
            #set net size
            nothing = 0
        if cell_params != 0:
            #set cell params
            nothing = 0
        agent_pop[i][j] = np.random.uniform(x_centre-(x_range/2), x_centre+(x_range/2))
        j += 1
        agent_pop[i][j] = np.random.uniform(y_centre-(y_range/2), y_centre+(y_range/2))
        j += 1
        agent_pop[i][j] = np.random.uniform(angle, angle_range)
        j += 1


# print agent_pop[3][3]
# print agent_pop[9][55]
# print agent_pop[5][47]

# #p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
# p.setup(timestep=1.0, min_delay=delay_min, max_delay=delay_max)
# #nNeurons = 20  # number of neurons in each population
# p.set_number_of_neurons_per_core(p.IF_curr_exp, 20)# / 2)

#cell configuration
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0,
                   'e_rev_E': 0.,
                   'e_rev_I': -80.
                   }


run_condition = Condition()
running = True

#I/O conditions
def send_spike(label, sender):
    running = True
    run_condition.acquire()
    if running:
        run_condition.release()
        sender.send_spike(label, 0, send_full_keys=True)
    else:
        run_condition.release()
        #break

def stop_flow(label, sender):
    run_condition.acquire()
    running = False
    run_condition.release()

motor_spikes = [0 for i in range(4)]
def receive_spikes(label, time, neuron_ids):
    for neuron_id in neuron_ids:
        run_condition.acquire()
        print "Received spike at time {} from {} - {}".format(time, label, neuron_id)
        run_condition.release()
        #add handler to process spike to motor command/ location change
        if label == neuron_labels[agent_neurons-4]:
            motor_spikes[0] += 1
            print "motor 0"
        elif label == neuron_labels[agent_neurons-3]:
            motor_spikes[1] += 1
            print "motor 1"
        elif label == neuron_labels[agent_neurons-2]:
            motor_spikes[2] += 1
            print "motor 2"
        elif label == neuron_labels[agent_neurons-1]:
            motor_spikes[3] += 1
            print "motor 3"
        else:
            print "failed motor receive"

def update_location(agent):
    print "before = ", agent_pop[agent][genetic_length-3], agent_pop[agent][genetic_length-2], agent_pop[agent][genetic_length-1]
    total_left = motor_spikes[0] #- motor_spikes[1]
    total_right = motor_spikes[2] #- motor_spikes[3]
    motor_average = (total_right + total_left) / 2.
    if total_left != 0 or total_right != 0:
        left_ratio = np.abs(float(total_left)/(np.abs(float(total_left))+np.abs(float(total_right))))
        right_ratio = np.abs(float(total_right)/(np.abs(float(total_left))+np.abs(float(total_right))))
    else:
        left_ratio = 0
        right_ratio = 0
    distance_moved = (left_ratio*total_right) + (right_ratio*total_left)
    print "left = {} <- {} + {}".format(total_left, motor_spikes[0], motor_spikes[1])
    print "right = {} <- {} + {}".format(total_right, motor_spikes[2], motor_spikes[3])
    #x - negative angle due to x y being opposite to trigonometry
    print "dx = ", (distance_moved * np.sin(-agent_pop[agent][genetic_length-1]))
    print "average = ", (total_left + total_right) / 2.
    print "new distance calc = ", distance_moved
    print np.sin(-agent_pop[agent][genetic_length-1])
    #y
    print "dy = ", (distance_moved * np.cos(-agent_pop[agent][genetic_length-1]))
    print np.cos(-agent_pop[agent][genetic_length-1])
    #angle
    print "change = ", (total_right - total_left) * 0.01
    angle_before = agent_pop[agent][genetic_length-1]
    agent_pop[agent][genetic_length-1] += (total_right - total_left) * 0.01
    if agent_pop[agent][genetic_length-1] > np.pi:
        agent_pop[agent][genetic_length - 1] -= np.pi * 2
    if agent_pop[agent][genetic_length-1] < -np.pi:
        agent_pop[agent][genetic_length - 1] += np.pi * 2
    #possbily change to move between half the angle of start and finish
    agent_pop[agent][genetic_length-3] += distance_moved * np.sin(-(angle_before+agent_pop[agent][genetic_length-1])/2)
    agent_pop[agent][genetic_length-2] += distance_moved * np.cos(-(angle_before+agent_pop[agent][genetic_length-1])/2)
    print "after = ", agent_pop[agent][genetic_length-3], agent_pop[agent][genetic_length-2], agent_pop[agent][genetic_length-1]
    for i in range(4):
        motor_spikes[i] = 0

def my_tan(dx, dy):
    theta = np.arctan(dy / dx)
    if dx < 0:
        theta -= np.pi
    theta -= np.pi / 2
    if theta < 0:
        theta += np.pi * 2
    if theta > np.pi:
        theta -= np.pi * 2
    return theta

def poisson_rate(agent, light_dist, light_angle):
    agent_x = agent_pop[agent][genetic_length-3]
    agent_y = agent_pop[agent][genetic_length-2]
    agent_angle = agent_pop[agent][genetic_length-1]
    #theta between pi and -pi relative to north anticlockwise positive
    light_x = light_dist * np.sin(-light_angle)
    light_y = light_dist * np.cos(-light_angle)
    theta = my_tan(light_x-agent_x, light_y-agent_y)
    #calculate and cap distance
    distance_cap = 200
    distance = np.sqrt(np.power(agent_x-light_x,2)+np.power(agent_y-light_y,2))
    if distance < distance_cap:
        distance = distance_cap
    #generate angle differnce between agent view and light location
    relative_view = theta - agent_angle
    if relative_view > np.pi:
        relative_view -= 2*np.pi
    if relative_view < -np.pi:
        relative_view += 2*np.pi
    #view bins
    bin_size = visual_field/visual_discrete
    sensor_reading = [0 for j in range(visual_discrete)]
    sensor_poisson = [0 for j in range(visual_discrete)]
    for i in range(visual_discrete):
        bin_angle = -(visual_field/2) + (i*bin_size)
        if relative_view > bin_angle and relative_view < (bin_angle+bin_size):
            sensor_reading[i] = 1
        else:
            #possibly wrong for certain values - maybe not anymore
            right_angle = relative_view-(bin_angle+bin_size)
            left_angle = relative_view-bin_angle
            if right_angle > np.pi:
                right_angle -= 2*np.pi
            if right_angle < -np.pi:
                right_angle += 2*np.pi
            if left_angle > np.pi:
                left_angle -= 2*np.pi
            if left_angle < -np.pi:
                left_angle += 2*np.pi
            min_angle = min(abs(left_angle), abs(right_angle))
            sensor_reading[i] = 1 - (min_angle/np.pi)
        if distance > distance_cap:
            sensor_poisson[i] = sensor_reading[i] * (np.power(distance_cap,2)/np.power(distance,2)) * max_poisson
        else:
            sensor_poisson[i] = sensor_reading[i] * max_poisson

    return sensor_poisson

def calc_instant_fitness(agent, light_dist, light_angle):
    agent_x = agent_pop[agent][genetic_length-3]
    agent_y = agent_pop[agent][genetic_length-2]
    light_x = light_dist * np.sin(-light_angle)
    light_y = light_dist * np.cos(-light_angle)
    fitness = np.sqrt(np.power(agent_x-light_x,2)+np.power(agent_y-light_y,2))
    return fitness

def reset_agent(agent):
    agent_pop[agent][genetic_length-1] = 0
    agent_pop[agent][genetic_length-2] = 0
    agent_pop[agent][genetic_length-3] = 0

def agent_fitness(agent, light_distance, light_theta, print_move):
    global port_offset
    global number_of_runs
    global counter
    print "\n\nStarting agent - {}\n\n".format(agent)
    p.setup(timestep=1.0, min_delay=delay_min, max_delay=delay_max)
    p.set_number_of_neurons_per_core(p.IF_curr_exp, 20)
    # setup of different neuronal populations
    #neuron_pop = list();
    neuron_pop = []
    if agent != 0:
        for i in range(agent_neurons):
            del neuron_labels[0]
    inhibitory_count = 0
    excitatory_count = 0
    for i in range(agent_neurons):
        if agent_pop[agent][inhibitory_loc + i] == -1:
            neuron_labels.append("Inhibitory{}-neuron{}-agent{}-port{}".format(inhibitory_count,i,agent,port_offset))
            neuron_pop.append(
                p.Population(neuron_pop_size, p.IF_curr_exp(), label=neuron_labels[i]))
            inhibitory_count += 1
        else:
            neuron_labels.append("Excitatory{}-neuron{}-agent{}-port{}".format(excitatory_count,i,agent,port_offset))
            neuron_pop.append(
                p.Population(neuron_pop_size, p.IF_curr_exp(), label=neuron_labels[i]))
            excitatory_count += 1
        # if print_move == True:
        #     neuron_pop[i].record(["spikes", "v"])

    # connect neuronal population according to genentic instructions
    projection_list = list()
    for i in range(agent_neurons):
        for j in range(agent_neurons):
            # if theres a connection connect
            if agent_pop[agent][set2loc + (i * agent_neurons) + j] != 0:
                # if connection is inhibitory set as such
                if agent_pop[agent][inhibitory_loc + i] == -1:
                    synapse = p.StaticSynapse(
                        weight=-agent_pop[agent][(i * agent_neurons) + j],
                        delay=agent_pop[agent][delay_loc + ((i * agent_neurons) + j)])
                    projection_list.append(p.Projection(
                        neuron_pop[i], neuron_pop[j], p.AllToAllConnector(),
                        synapse, receptor_type="inhibitory"))
                # set as excitatory
                else:
                    synapse = p.StaticSynapse(
                        weight=agent_pop[agent][(i * agent_neurons) + j],
                        delay=agent_pop[agent][delay_loc + ((i * agent_neurons) + j)])
                    projection_list.append(p.Projection(
                        neuron_pop[i], neuron_pop[j], p.AllToAllConnector(),
                        synapse, receptor_type="excitatory"))
                    # set STDP, weight goes to negative if inhibitory?
                    # stdp_model = p.STDPMechanism(
                    #     timing_dependence=p.SpikePairRule(
                    #         tau_plus=20., tau_minus=20.0, A_plus=0.5, A_minus=0.5),
                    #         weight_dependence=p.AdditiveWeightDependence(w_min=weight_min, w_max=weight_max))

    # connect in and out live links
    #visual_input = list()
    visual_input = []
    visual_projection = []#list()
    input_labels = []#list()
    #sensor_poisson = [0 for j in range(visual_discrete)]
    sensor_poisson = poisson_rate(agent, 200, np.pi / 4)
    for i in range(visual_discrete):
        print i
        input_labels.append("input_spikes{}".format(i))
        visual_input.append(p.Population(
            1, p.SpikeSourcePoisson(rate=sensor_poisson[i]), label=input_labels[i]))
        visual_projection.append(p.Projection(
            visual_input[i], neuron_pop[i], p.OneToOneConnector(), p.StaticSynapse(
                weight=visual_weight, delay=visual_delay)))
    # for i in range(4):
    #     del motor_labels[0]
    motor_labels = []
    for i in range(4):
        print i
        motor_labels.append(neuron_labels[agent_neurons - (i + 1)])
        p.external_devices.activate_live_output_for(neuron_pop[agent_neurons - (i + 1)], database_notify_port_num=19800+port_offset)
    live_connection = p.external_devices.SpynnakerLiveSpikesConnection(
        receive_labels=[motor_labels[0], motor_labels[1], motor_labels[2],motor_labels[3]], local_port=(19800+port_offset))
    for i in range(4):
        live_connection.add_receive_callback(motor_labels[i], receive_spikes)
    fitness = 0
    # spikes = list()
    # v = list()
    print"\nstarting run\n"
    for i in range(0,total_runtime, time_slice):
        p.run(time_slice)
        update_location(agent)
        fitness += calc_instant_fitness(agent, light_distance, light_theta)
        sensor_poisson = poisson_rate(agent, light_distance, light_theta)
        for j in range(visual_discrete):
            visual_input[j].set(rate=sensor_poisson[j])
        print "did a run {}/{}, time now at {}/{} and fitness = {}".format(counter, number_of_runs, i+time_slice, total_runtime, fitness)
        if print_move == True:
            with open('movement {}.csv'.format((port_offset-counter)/number_of_runs), 'a') as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                writer.writerow([agent_pop[agent][genetic_length-3],agent_pop[agent][genetic_length-2],agent_pop[agent][genetic_length-1]])
    # if print_move == True:
    #     spikes = []
    #     v = []
    #     for j in range(agent_neurons):
    #         spikes.append(neuron_pop[j].get_data("spikes"))
    #         v.append(neuron_pop[j].get_data("v"))
    live_connection.close()
    live_connection._handle_possible_rerun_state()
    p.end()
    if counter != number_of_runs:
        counter += 1
        port_offset += 1
        reset_agent(agent)
        if print_move == True:
            with open('movement {}.csv'.format((port_offset-counter)/number_of_runs), 'a') as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                writer.writerow([light_distance*np.sin(light_theta), light_distance*np.cos(light_theta)])
        port_recurse_check = port_offset
        fitness += agent_fitness(agent, light_distance, -light_theta, print_move)
    else:
        counter = 1
        port_offset -= 1
    port_offset += 1
    reset_agent(agent)
    return fitness

def bubble_sort_fitness(fitnesses):
    order = [i for i in range(pop_size)]
    for i in range(pop_size):
        for j in range(pop_size-i-1):
            if fitnesses[order[j]] > fitnesses[order[j+1]]:
                temp = order[j+1]
                order[j+1] = order[j]
                order[j] = temp
    return order

def mate_agents(mum, dad):
    for i in range(genetic_length-3):
        if np.random.uniform(0, 1) < 0.5:
            agent_pop[child][i] = agent_pop[mum][i]
        else:
            agent_pop[child][i] = agent_pop[dad][i]
        if np.random.uniform(0, 1) < mutation_rate:
            if i < weights:
                if recurrency == 0 and i%(agent_neurons+1) == 0:
                    agent_pop[child][i] = 0;
                else:
                    agent_pop[child][i] = np.random.normal(agent_pop[child][i], weight_range*shift_ratio)
                    if agent_pop[child][i] < weight_min:
                        agent_pop[child][i] += weight_range
                    if agent_pop[child][i] > weight_max:
                        agent_pop[child][i] -= weight_range
                    if agent_pop[child][i] < weight_cut:
                        agent_pop[child][i] = 0
            elif i < weights + delays:
                if recurrency == 0 and (i-1)%(agent_neurons+1):
                    agent_pop[child][i] = 0;
                else:
                    agent_pop[child][i] = np.random.normal(agent_pop[child][i], delay_range*shift_ratio)
                if agent_pop[child][i] < delay_min:
                    agent_pop[child][i] += delay_range
                if agent_pop[child][i] > delay_max:
                    agent_pop[child][i] -= delay_range
            elif inhibitory != 0 and i < inhibitory_loc + agent_neurons:
                agent_pop[child][i] *= -1
            elif i < set2loc + set2zero:
                if agent_pop[child][i] == 1:
                    agent_pop[child][i] = 0
                else:
                    agent_pop[child][i] = 1
            elif i < set2loc + set2zero + 1:
                if np.random.uniform(0, 1) < plastic_prob:
                    agent_pop[child][i] = 0
                else:
                    agent_pop[child][i] = 1
            else:
                print "shouldn't be here, location saved for further genetic manipulation"
    agent_pop[child][i] = np.random.uniform(x_centre - (x_range / 2), x_centre + (x_range / 2))
    i += 1
    agent_pop[child][i] = np.random.uniform(y_centre - (y_range / 2), y_centre + (y_range / 2))
    i += 1
    agent_pop[child][i] = np.random.uniform(angle, angle_range)
    #return child

def copy_child(location, temp):
    if temp == False:
        for i in range(genetic_length):
            agent_pop[location][i] = agent_pop[child][i]
    else:
        for i in range(genetic_length):
            temp_pop[location][i] = agent_pop[child][i]

def start_new_gen(temp, pop):
    for i in range(pop_size):
        pop[i] = temp[i]
        temp[i] = 0
        for j in range(genetic_length):
            agent_pop[i][j] = temp_pop[i][j]
            temp_pop[i][j] = 0

#port definitions
cell_params_spike_injector = {
    'port': 19996,
}

temp_fitness = [0 for i in range(pop_size)]

cell_params_spike_injector_with_key = {
    'port': 12346,
    'virtual_key': 0x70000,
}
if seed_population == True and copy_population == True:
    print "No need to test all of the population again"
else:
    pop_fitness = [0 for i in range(pop_size)]
    total_fitness = 0
    #calculate fitnesses for all agents in population
    for agent in range(pop_size):
    #     with open('movement {}.csv'.format((port_offset-counter)/number_of_runs), 'w') as file:
    #         writer = csv.writer(file, delimiter=',', lineterminator='\n')
    #         writer.writerow([200*np.sin(np.pi / 4), 200*np.cos(np.pi / 4)])
    #         writer.writerow([agent_pop[agent][genetic_length - 3], agent_pop[agent][genetic_length - 2],
    #                          agent_pop[agent][genetic_length - 1]])
        pop_fitness[agent] = agent_fitness(agent, 200, -np.pi/4, False)
        total_fitness += pop_fitness[agent]
        print"\n\n Terminated agent - {} \n\n".format(agent)
    #total_v = list()
    # for i in range(0,total_runtime, time_slice):
    #     p.run(time_slice)
    #     # for j in range(agent_neurons):
    #     #     spikes.append(neuron_pop[j].get_data("spikes"))
    #     #     v.append(neuron_pop[j].get_data("v"))
    #     # Figure(
    #     #     # raster plot of the presynaptic neuron spike times
    #     #     Panel(spikes[0+((i/time_slice)*agent_neurons)].segments[0].spiketrains,
    #     #           yticks=True, markersize=2, xlim=(0, i+time_slice)),
    #     #     Panel(spikes[1+((i/time_slice)*agent_neurons)].segments[0].spiketrains,
    #     #           yticks=True, markersize=2, xlim=(0, i+time_slice)),
    #     #     Panel(spikes[2+((i/time_slice)*agent_neurons)].segments[0].spiketrains,
    #     #           yticks=True, markersize=2, xlim=(0, i+time_slice)),
    #     #     Panel(spikes[3+((i/time_slice)*agent_neurons)].segments[0].spiketrains,
    #     #           yticks=True, markersize=2, xlim=(0, i+time_slice)),
    #     #     Panel(spikes[4+((i/time_slice)*agent_neurons)].segments[0].spiketrains,
    #     #           yticks=True, markersize=2, xlim=(0, i+time_slice)),
    #     #     title="Simple synfire chain example with injected spikes",
    #     #     annotations="Simulated with {}".format(p.name())
    #     # )
    #     #plt.show()
    #     # Figure(
    #     #     Panel(v[0+((i/time_slice)*agent_neurons)].segments[0].filter(name='v')[0],
    #     #           ylabel="Membrane potential (mV)", yticks=True, xlim=(0, i+time_slice)),
    #     #     Panel(v[1+((i/time_slice)*agent_neurons)].segments[0].filter(name='v')[0],
    #     #           ylabel="Membrane potential (mV)", yticks=True, xlim=(0, i+time_slice)),
    #     #     Panel(v[2+((i/time_slice)*agent_neurons)].segments[0].filter(name='v')[0],
    #     #           ylabel="Membrane potential (mV)", yticks=True, xlim=(0, i+time_slice)),
    #     #     Panel(v[3+((i/time_slice)*agent_neurons)].segments[0].filter(name='v')[0],
    #     #           ylabel="Membrane potential (mV)", yticks=True, xlim=(0, i+time_slice)),
    #     #     Panel(v[4+((i/time_slice)*agent_neurons)].segments[0].filter(name='v')[0],
    #     #           ylabel="Membrane potential (mV)", yticks=True, xlim=(0, i+time_slice)),
    #     #     # splot.SpynakkerPanel(v[0].segments[0].filter(name='v')[0],
    #     #     #                      ylabel="Membrane potential (mV) 0", yticks=True, xlim=(0, i+time_slice+time_slice)),
    #     #     # splot.SpynakkerPanel(v[1].segments[0].filter(name='v')[0],
    #     #     #                      ylabel="Membrane potential (mV) 1", yticks=True, xlim=(0, i+time_slice+time_slice)),
    #     #     # splot.SpynakkerPanel(cd,
    #     #     #                      ylabel="Membrane potential (mV) 2", yticks=True, xlim=(0, i+time_slice+time_slice)),
    #     #     # splot.SpynakkerPanel(v[3].segments[0].filter(name='v')[0],
    #     #     #                      ylabel="Membrane potential (mV) 3", yticks=True, xlim=(i, i+time_slice)),
    #     #     # splot.SpynakkerPanel(v[4].segments[0].filter(name='v')[0],
    #     #     #                      ylabel="Membrane potential (mV) 4", yticks=True, xlim=(i, i+time_slice)),
    #     # )
    #     # plt.show()
    #     update_location(agent)
    #     pop_fitness[agent] += calc_fitness(agent, 200, np.pi / 4)
    #     sensor_poisson = poisson_rate(agent, 200, np.pi / 4)
    #     for j in range(visual_discrete):
    #         visual_input[j].set(rate=sensor_poisson[j])
    #     print "did a run"
with open('Fitness over time.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n')
    writer.writerow(pop_fitness)
if only_improve == True:
    with open('Fitness improvement record.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow(pop_fitness)
else:
    with open('Child fitness record.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow(pop_fitness)
#generate new pop based on fitness evaluations
#sort fitness values
order = bubble_sort_fitness(pop_fitness)
worst_fitness = pop_fitness[order[pop_size-1]]
total_fitness = fitness_offset + (worst_fitness*pop_size) - total_fitness
for count in range(number_of_children):
    i = np.random.uniform(0,total_fitness)
    j = 0
    #generate parents to mate
    print "i = ",i,
    while i > 0:
        i -= fitness_offset + worst_fitness - pop_fitness[order[j]]
        j += 1
    mum = order[j-1]
    print "mum = {} from a j of {}".format(mum, j)
    # dad = mum possibly useful to stop self replication polluting the gene pool
    # while dad == mum:
    i = np.random.uniform(0,total_fitness)
    j = 0
    print "i = ",i,
    while i > 0:
        i -= fitness_offset + worst_fitness - pop_fitness[order[j]]
        j += 1
    dad = order[j-1]
    print "dad = {} from a j of {}".format(dad, j)
    #generate child
    mate_agents(mum, dad)
    with open('movement {}.csv'.format((port_offset-counter)/number_of_runs), 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow([200*np.sin(np.pi / 4), 200*np.cos(np.pi / 4)])
        writer.writerow([agent_pop[agent][genetic_length - 3], agent_pop[agent][genetic_length - 2],
                         agent_pop[agent][genetic_length - 1]])
    child_fitness = agent_fitness(child, 200, -np.pi/4, True)
    print "generated a child ",
    if only_improve == True:
        if child_fitness < pop_fitness[order[pop_size-1]]:
            print "which was then added to the population"
            copy_child(order[pop_size-1], False)
            #agent_pop[order[pop_size-1]] = agent_pop[child] ## check this works right
            pop_fitness[order[pop_size-1]] = child_fitness
            worst_positon = order[pop_size-1]
            order = bubble_sort_fitness(pop_fitness)
            worst_fitness = pop_fitness[order[pop_size-1]]
            total_fitness = 0
            for i in range(pop_size):
                total_fitness += fitness_offset + worst_fitness - pop_fitness[i]
            with open('Fitness improvement record.csv', 'a') as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                writer.writerow([(port_offset-1)/number_of_runs, worst_positon, child_fitness])
        with open('Fitness over time.csv', 'a') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            writer.writerow(pop_fitness)
    else:
        temp_fitness[count%pop_size] = child_fitness
        copy_child(count%pop_size, True)
        with open('Child fitness record.csv', 'a') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            writer.writerow([count%pop_size, child_fitness])
        if count%pop_size == pop_size-1:
            start_new_gen(temp_fitness, pop_fitness)
            order = bubble_sort_fitness(pop_fitness)
            worst_fitness = pop_fitness[order[pop_size-1]]
            total_fitness = 0
            with open('Fitness over time.csv', 'a') as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                writer.writerow(pop_fitness)
            for i in range(pop_size):
                total_fitness += fitness_offset + worst_fitness - pop_fitness[i]

with open('population_genes.csv', 'w') as file:
    writer = csv.writer(file)
    for i in range(pop_size):
        writer.writerow([pop_fitness[order[i]],agent_pop[order[i]]])

with open('movement {}.csv'.format((port_offset-counter)/number_of_runs), 'w') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n')
    writer.writerow([200*np.sin(-np.pi / 4), 200*np.cos(-np.pi / 4)])
    writer.writerow([agent_pop[order[0]][genetic_length - 3], agent_pop[order[0]][genetic_length - 2],
                     agent_pop[order[0]][genetic_length - 1]])

best_fitness = agent_fitness(order[0], 200, -np.pi/4, True)
best_fitness2 = agent_fitness(order[0], 200, -np.pi/4, True)

print "\n shit finished yo!!"