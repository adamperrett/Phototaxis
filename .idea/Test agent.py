import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import time
import pylab
import numpy as np
from threading import Condition
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot
import matplotlib.animation as live
from matplotlib import gridspec
from pyNN.random import RandomDistribution as rand
import spynnaker8.spynakker_plotting as splot
import csv
import pandas

#run setup
seed_population = True
copy_population = False
only_improve = False
total_runtime = 2000
time_slice = 100
pop_size = 20
reset_count = 10
no_move_punishment = 2.
agent_neurons = 6
neuron_pop_size = 1
ex_in_ratio = 4#:1
visual_discrete = 2
visual_field = (2./6.)*np.pi
max_poisson = 300
mutation_rate = 0.02
shift_ratio = 0.2
number_of_children = 200
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
seed_from = -27
seed_network = [0 for i in range(genetic_length)] #[pop][gen]
with open('Created network.csv') as from_file:
    csvFile = csv.reader(from_file)
    for row in csvFile:
        temp = row
        if abs(float(temp[0]) - seed_from) < 1e-10:
            for j in range(genetic_length):
                seed_network[j] = float(temp[j+3])
            break

inhibitory_loc = 72
set2loc = 78
delay_loc = 36

fig = plt.figure()
tracking = fig.add_subplot(1,1,1)
xs = []
ys = []
ld = 20
lt = 3

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
    print "before = ", seed_network[genetic_length-3], seed_network[genetic_length-2], seed_network[genetic_length-1]
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
    print "dx = ", (distance_moved * np.sin(-seed_network[genetic_length-1]))
    print "average = ", (total_left + total_right) / 2.
    print "new distance calc = ", distance_moved
    print np.sin(-seed_network[genetic_length-1])
    #y
    print "dy = ", (distance_moved * np.cos(-seed_network[genetic_length-1]))
    print np.cos(-seed_network[genetic_length-1])
    #angle
    print "change = ", (total_right - total_left) * 0.01
    angle_before = seed_network[genetic_length-1]
    seed_network[genetic_length-1] += (total_right - total_left) * 0.01
    if seed_network[genetic_length-1] > np.pi:
        seed_network[genetic_length - 1] -= np.pi * 2
    if seed_network[genetic_length-1] < -np.pi:
        seed_network[genetic_length - 1] += np.pi * 2
    #possbily change to move between half the angle of start and finish
    seed_network[genetic_length-3] += distance_moved * np.sin(-(angle_before+seed_network[genetic_length-1])/2)
    seed_network[genetic_length-2] += distance_moved * np.cos(-(angle_before+seed_network[genetic_length-1])/2)
    print "after = ", seed_network[genetic_length-3], seed_network[genetic_length-2], seed_network[genetic_length-1]
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
    agent_x = seed_network[genetic_length-3]
    agent_y = seed_network[genetic_length-2]
    agent_angle = seed_network[genetic_length-1]
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
    agent_x = seed_network[genetic_length-3]
    agent_y = seed_network[genetic_length-2]
    light_x = light_dist * np.sin(-light_angle)
    light_y = light_dist * np.cos(-light_angle)
    fitness = np.sqrt(np.power(agent_x-light_x,2)+np.power(agent_y-light_y,2))
    return fitness

def reset_agent(agent):
    seed_network[genetic_length-1] = 0
    seed_network[genetic_length-2] = 0
    seed_network[genetic_length-3] = 0

def animate(i):
    global lt, ld, xs, ys
    tracking.clear()
    tracking.plot(ld * np.sin(-lt), ld * np.cos(-lt), 'ro')
    tracking.plot(xs, ys, '-')
    tracking.set_xlim(-200, 200)
    tracking.set_ylim(0, 200)
    #plt.show(block=False)

def agent_fitness(agent, light_distance, light_theta, print_move):
    global port_offset
    global number_of_runs
    global counter
    global ld, lt, xs, ys
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
    #initialise neural populations
    for i in range(agent_neurons):
        if seed_network[inhibitory_loc + i] == -1:
            neuron_labels.append("Inhibitory{}-neuron{}-agent{}-port{}".format(inhibitory_count,i,agent,port_offset))
            neuron_pop.append(
                p.Population(neuron_pop_size, p.IF_cond_exp(), label=neuron_labels[i]))
            inhibitory_count += 1
        else:
            neuron_labels.append("Excitatory{}-neuron{}-agent{}-port{}".format(excitatory_count,i,agent,port_offset))
            neuron_pop.append(
                p.Population(neuron_pop_size, p.IF_cond_exp(), label=neuron_labels[i]))
            excitatory_count += 1
        # if print_move == True:
        #     neuron_pop[i].record(["spikes", "v"])

    # connect neuronal population according to genentic instructions
    projection_list = list()
    for i in range(agent_neurons):
        for j in range(agent_neurons):
            # if theres a connection connect
            if seed_network[set2loc + (i * agent_neurons) + j] != 0:
                # if connection is inhibitory set as such
                if seed_network[inhibitory_loc + i] == -1:
                    synapse = p.StaticSynapse(
                        weight=-seed_network[(i * agent_neurons) + j],
                        delay=seed_network[delay_loc + ((i * agent_neurons) + j)])
                    projection_list.append(p.Projection(
                        neuron_pop[i], neuron_pop[j], p.AllToAllConnector(),
                        synapse, receptor_type="inhibitory"))
                # set as excitatory
                else:
                    synapse = p.StaticSynapse(
                        weight=seed_network[(i * agent_neurons) + j],
                        delay=seed_network[delay_loc + ((i * agent_neurons) + j)])
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
    xs.append(seed_network[genetic_length-3])
    #location_x.append(47)
    ys.append(seed_network[genetic_length-2])
    #location_y.append(150)
    ld = light_distance
    lt = light_theta
    #live_graph = live.FuncAnimation(fig, animate, interval=2000, blit=True)
    #plt.show(block=False)

    #plt.show()
    for i in range(0,total_runtime, time_slice):
        p.run(time_slice)
        update_location(agent)
        xs.append(seed_network[genetic_length-3])
        ys.append(seed_network[genetic_length-2])
        tracking.clear()
        tracking.plot(ld * np.sin(-lt), ld * np.cos(-lt), 'ro')
        tracking.plot(xs, ys, '-')
        # tracking.set_xlim(-200, 200)
        # tracking.set_ylim(0, 200)
        plt.show(block=False)
        sensor_poisson = poisson_rate(agent, light_distance, light_theta)
        for j in range(visual_discrete):
            visual_input[j].set(rate=sensor_poisson[j])
        print "did a run {}/{}, time now at {}/{} and fitness = {}/{}".format\
            (counter, number_of_runs, i+time_slice, total_runtime, fitness, light_distance*((i/time_slice)+1))
        if print_move == True:
            with open('movement of {}.csv'.format(seed_from), 'a') as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                writer.writerow([seed_network[genetic_length-3],seed_network[genetic_length-2],seed_network[genetic_length-1]])
    # if print_move == True:
    #     spikes = []
    #     v = []
    #     for j in range(agent_neurons):
    #         spikes.append(neuron_pop[j].get_data("spikes"))
    #         v.append(neuron_pop[j].get_data("v"))
    live_connection.close()
    live_connection._handle_possible_rerun_state()
    port_offset += 1
    reset_agent(0)
    p.end()
    return fitness

#port definitions
cell_params_spike_injector = {
    'port': 19996,
}

cell_params_spike_injector_with_key = {
    'port': 12346,
    'virtual_key': 0x70000,
}

with open('movement of {}.csv'.format(seed_from), 'w') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n')
    writer.writerow([0,0,0])
for i in range(5):
    with open('movement of {}.csv'.format(seed_from), 'a') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow([0, 0, 0])
    agent_fitness(0,200,np.pi/4, True)
with open('movement of {}.csv'.format(seed_from), 'a') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n')
    writer.writerow([-141, 141])
for i in range(5):
    with open('movement of {}.csv'.format(seed_from), 'a') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow([0, 0, 0])
    agent_fitness(0,200,-np.pi/4, True)

print "\n shit finished yo!!"