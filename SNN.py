import numpy as np
from sklearn.datasets import fetch_openml
import sys

"""
THIS CODE TRAINS AND TESTS A SPIKING NEURAL NETWORK
COMMAND LINE ARGUMENTS:
$1: number of neurons in the network
$2: theta value for neurons
"""

class InhibitedNeuronLayer:
    """
    Instead of a single neuron, this class represents a layer of N neurons
    This layer is inhibited - the spiking of a single neuron sets the voltages of all other neurons to the reset value
    Each neuron has a voltage, input current, voltage threshold, threshold increment theta, theta decay rate, and spiking value
    These are stored in arrays
    """
    def __init__(self, N, tau=100, v_reset=0.0, v_thresh=500, theta=10.0, tau_theta=0.001):
        self.size = N                                   # Size of the layer
        self.tau = tau                                # Decay rate
        self.v_reset = v_reset                          # Reset potential after spike
        self.v_thresh0 = v_thresh                       # Relaxed threshold potential
        self.v_thresh = np.full(N,v_thresh,dtype=float) # Threshold potentials
        self.theta = theta                              # Amount to increment threshold potential after spike
        self.tau_theta = tau_theta                      # Theta decay rate
        self.v = np.full(N,v_reset,dtype=float)         # Initial potentials
        self.spike = np.full(N,False)                   # Spike flags (0|1)

    def step(self, input_currents):
        """Simulate one time step with input currents.
            The input current shoud be a 1d numpy array with shape N"""
        self.v += -self.v/self.tau + input_currents

        # Check for spike
        self.spike = self.v >= self.v_thresh                           #spike if voltage is greater than threshold voltage
        if np.any(self.spike):
            self.v[:] = self.v_reset # set voltage of all neurons to reset voltage for spike event
            self.v_thresh[self.spike==True] += self.theta                  #alter the voltage threshold for any neurons that have just spiked
        self.v_thresh += (self.v_thresh0-self.v_thresh)*self.tau_theta #relax the voltage threshold back to the initial value

        return self.v, self.spike

    def static_step(self, input_currents):
        """Simulate one time step with input currents.
            The input current shoud be a 1d numpy array with shape N
            This is the same as the 'step' function but does not update the voltage thresholds of any of the neurons"""
        self.v += -self.v/self.tau + input_currents

        # Check for spike
        self.spike = self.v >= self.v_thresh                           #spike if voltage is greater than threshold voltage
        if np.any(self.spike):
            self.v[:] = self.v_reset # set voltage of all neurons to reset voltage for spike event

        return self.v, self.spike

    def reset(self):
        """Reset neuron voltages to their resting values after a digit has been presented
        """
        self.v[:] = self.v_reset
        return None

class SynapseLayer:
    """
    This represents a layer of synapses connecting two different layers of neurons
    Synaptic weights update according to the STDP rule
    The synapse layer has an input size and output size
    Individual synapses have a learning rates, a minumum and maximum weight value, and a spike duration
    They exist as a matrix of weights of dimensions input_size X output_size
    Initial weights are set to random values around half the max weight
    """
    def __init__(self, input_size, output_size, alpha_plus, alpha_minus, beta, min_weight=0, max_weight=1,spike_duration=25):
        self.min_weight = min_weight                                                     # Minumum synaptic weight
        self.max_weight = max_weight                                                     # Maximum synaptic weight
        self.weights = np.clip(np.random.normal(0,0.25,(input_size,output_size))+max_weight/2,min_weight,max_weight) # Initialize layer weights
        self.beta = beta                                                                 # STDP exponential factor
        self.input_size = input_size                                                     # Size of input
        self.output_size = output_size                                                   # Size of output
        self.input_spike_trace = np.zeros(self.input_size,dtype=int)        # spike traces
        self.alpha_plus = alpha_plus                                                     # amount to increase synaptic weights
        self.alpha_minus = alpha_minus #amount to decrease synaptic weights
        self.spike_trace_ages = np.zeros(self.input_size,dtype=int)        # age of spike traces
        self.spike_duration = spike_duration #duration of spike

    def update_weights(self, input_spikes, output_spikes):
        """Simulate one time step and any subsequent weight updates
           Takes pre and post synaptic spiking data as inputs to update pre and postsynaptic traces as well as weights"""
        #handle spiking events and updating spike traces
        self.spike_trace_ages[:]+=1
        self.input_spike_trace[self.spike_trace_ages>self.spike_duration]=0
        self.input_spike_trace[input_spikes==True]=1
        self.spike_trace_ages[input_spikes==True]=0
        # print(self.input_spike_trace)
        # print(self.weights)
        # print(self.weights[self.input_spike_trace==1,output_spikes==True])
        # print(self.weights[self.input_spike_trace==0,output_spikes==True])
        modify_up=np.outer(self.input_spike_trace==True,output_spikes).astype(bool)
        self.weights[modify_up]+=self.alpha_plus*np.exp(-self.beta*(self.weights[modify_up]-self.min_weight)/(self.max_weight-self.min_weight))
        modify_down=np.outer(self.input_spike_trace==False,output_spikes).astype(bool)
        # print(modify_up)
        # print(modify_down)
        self.weights[modify_down]+=self.alpha_minus*np.exp(-self.beta*(self.max_weight-self.weights[modify_down])/(self.max_weight-self.min_weight))
        
        self.weights=np.clip(self.weights,self.min_weight,self.max_weight) #make sure weights are between min and max values
        return None
        
    def step(self):
        """Simulate one time step without any weight updates"""
        #handle updating spike traces
        self.spike_trace_ages[:]+=1
        
        return None

    def reset(self):
        """Reset pre- and postsynaptic traces to their resting values after a digit has been presented
        """
        self.input_spike_trace[:] = 0
        self.spike_trace_ages[:] = 0
        return None
        
def train_round(neurons,synapses,training_data,training_labels,training_time=60000):
    """
    Training process for unsupervised learning of handwritten digits in a two-layer network
    Inputs:
        neurons: InhibitedNeuronLayer representing the output neurons of the network
        synapses: SynapseLayer representing connections between input and output neurons in the network
        training_data: the training data containing the handwritten digits
        training_labels: labels for the training data assigning each handwritten digit to its proper value
        training_time: the number of training examples to present
    """

    #do some training on the neurons
    digits_presented=np.zeros(10)
    digit_time=350
    for i in range(training_time):
        #voltages of neurons in the SNN
        outputvoltages = np.zeros((neurons.size,digit_time),dtype=float)
        #spiking behavior of neurons in the SNN
        outputspikes = np.zeros((neurons.size,digit_time))
        digits_presented[training_labels[i]]+=1
        inputs = np.abs(np.ravel(training_data[i])/4)
        inputs = np.reshape(inputs,(len(inputs),1)) #input data
        inputspikes=np.random.rand(inputs.shape[0],digit_time) #convert input data to spike trains
        inputspikes=inputspikes<inputs #convert input data to spike trains
        
        #feed in the spike train and allow the network to continuously update
        for j in range(digit_time):
            output_currents=(inputspikes[:,j].T@synapses.weights).T                 #calculate current to output neurons
            outputvoltages[:,j],outputspikes[:,j]=neurons.step(output_currents) #update voltages and spiking of output neurons
            if np.any(outputspikes[:,j]):
                synapses.update_weights(inputspikes[:,j],outputspikes[:,j])             #update synapses
            else:
                synapses.step() #no weight updates if there are no output spikes
        
        #after feeding one digit, let the network rest enough to allow the voltages and synaptic traces to decay back to their resting values
        neurons.reset()
        synapses.reset()

    return None
    
def train_epochs(neurons,synapses,training_data,training_labels,n_epochs):
    """
    Train the system for a number of full epochs
    Inputs:
        neurons: InhibitedNeuronLayer representing the output neurons of the network
        synapses: SynapseLayer representing connections between input and output neurons in the network
        training_data: the training data containing the handwritten digits
        training_labels: labels for the training data assigning each handwritten digit to its proper value
        n_epochs: the number of training epochs to present
    """
    for i in range(n_epochs):
        train_round(neurons,synapses,training_data,training_labels)
    
def label_outputs(neurons,synapses,training_data,training_labels):
    """
    Label the output neurons with the digits that cause them to spike the most
    Inputs:
        neurons: InhibitedNeuronLayer representing the output neurons of the network
        synapses: SynapseLayer representing connections between input and output neurons in the network
        training_data: the training data containing the handwritten digits
        training_labels: labels for the training data assigning each handwritten digit to its proper value
    Returns: array of length neurons.size identifying what digit each output neuron corresponds to
    """
    classification_time=10000
    spike_frequencies=np.zeros((neurons.size,10))
    digit_time=350
    digits_presented=np.zeros(10)
    for i in range(classification_time):
        #voltages of neurons in the SNN
        outputvoltages = np.zeros((neurons.size,digit_time),dtype=float)
        #spiking behavior of neurons in the SNN
        outputspikes = np.zeros((neurons.size,digit_time))
        digits_presented[training_labels[i]]+=1
        inputs = np.abs(np.ravel(training_data[i])/4)
        inputs = np.reshape(inputs,(len(inputs),1)) #input data
        inputspikes=np.random.rand(inputs.shape[0],digit_time) #convert input data to spike trains
        inputspikes=inputspikes<inputs #convert input data to spike trains
        
        #feed in the spike train and allow the network to continuously update
        for j in range(digit_time):
            output_currents=(inputspikes[:,j].T@synapses.weights).T                 #calculate current to output neurons
            outputvoltages[:,j],outputspikes[:,j]=neurons.static_step(output_currents) #update voltages and spiking of output neurons
            spike_frequencies[:,training_labels[i]]+=outputspikes[:,j]
            
        #after feeding one digit, let the network rest enough to allow the voltages and synaptic traces to decay back to their resting values
        neurons.reset()
    return np.argmax(spike_frequencies,axis=1)
    
def test_accuracy(neurons,synapses,training_data,training_labels,neuron_ids):
    """
    Tests the accuracy of the network in learning handwritten digits
    Inputs:
        neurons: InhibitedNeuronLayer representing the output neurons of the network
        synapses: SynapseLayer representing connections between input and output neurons in the network
        training_data: the training data containing the handwritten digits
        training_labels: labels for the training data assigning each handwritten digit to its proper value
        neuron_ids: array identifying which digit each neuron corresponds to
    Returns: accuracy, confusion matrix
    """
    #now estimate accuracy
    test_time=10000
    confusion_matrix=np.zeros((10,10))
    digit_time=350
    digits_presented=np.zeros(10)
    correct_predictions=0
    for i in range(test_time):
        #voltages of neurons in the SNN
        outputvoltages = np.zeros((neurons.size,digit_time),dtype=float)
        #spiking behavior of neurons in the SNN
        outputspikes = np.zeros((neurons.size,digit_time))
        digits_presented[training_labels[i+60000]]+=1
        inputs = np.abs(np.ravel(training_data[i+60000])/4)
        inputs = np.reshape(inputs,(len(inputs),1)) #input data
        inputspikes=np.random.rand(inputs.shape[0],digit_time) #convert input data to spike trains
        inputspikes=inputspikes<inputs #convert input data to spike trains
        
        #feed in the spike train and allow the network to continuously update
        for j in range(digit_time):
            output_currents=(inputspikes[:,j].T@synapses.weights).T                 #calculate current to output neurons
            outputvoltages[:,j],outputspikes[:,j]=neurons.static_step(output_currents) #update voltages and spiking of output neurons

        #determine the neuron with which this digit is most associated
        neuron_id_test=np.argmax(np.sum(outputspikes,axis=1))
        predicted_digit=neuron_ids[neuron_id_test]
        correct_digit=training_labels[i+60000]
        if predicted_digit==correct_digit:
            correct_predictions+=1
        confusion_matrix[predicted_digit,correct_digit]+=1
        
        #after feeding one digit, let the network rest enough to allow the voltages and synaptic traces to decay back to their resting values
        neurons.reset()

    accuracy=correct_predictions/test_time #determine accuracy
    confusion_matrix = confusion_matrix/digits_presented #normalize confusion matrix
    return accuracy, confusion_matrix
    
#do some testing
if __name__ == "__main__":
    
    #get mnist data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    training_data = mnist['data'].reshape(-1, 28, 28) / 255.0  # shape: (70000, 28, 28)
    training_labels = mnist['target'].astype(int)
    
    #create neurons and synapses
    neurons=InhibitedNeuronLayer(int(sys.argv[1]), tau=100, v_reset=0.0, v_thresh=500, theta=float(sys.argv[2]), tau_theta=0.0001)
    synapses=SynapseLayer(784,neurons.size,alpha_plus=0.01,alpha_minus=-0.005,beta=3,min_weight=0,max_weight=1,spike_duration=25)
    
    #do some fake training
    train_epochs(neurons,synapses,training_data,training_labels,10)
    
    #label the digits
    neuron_ids=label_outputs(neurons,synapses,training_data,training_labels)
    
    #test the network
    accuracy, confusion_matrix = test_accuracy(neurons,synapses,training_data,training_labels,neuron_ids)
    
    #save results to files
    np.savetxt(f'confusion_matrix_{sys.argv[1]}_{sys.argv[2]}.npy', confusion_matrix)
    np.savetxt(f'accuracy_{sys.argv[1]}_{sys.argv[2]}.txt',np.array([accuracy]))