import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Generalized cerebellar adaptive filter with alpha-like impulse responses as temporal basis functions and with a bank of Purkinje cells sensitive to a range of input-output delays.
class Cerebellum(object):

    # Creates a Module composed of Alpha-like filters with different temporal profiles.
    def __init__(self, dt=0.01, nInputs=1, nOutputs=1, nPCpop=10, nIndvBasis=50, nSharedBasis=200, beta_MF=1e-3, beta_PF=1e-6, 
                 range_delays=[0.05, 0.5], range_TC=[0.05, 2.], range_scaling=[1, 100], range_W=[0., 1.]):

        # Basic parameters.
        self.dt = dt
        self.nInputs = nInputs
        self.nIndvBasis = nIndvBasis
        self.nSharedBasis = nSharedBasis
        self.nOutputs = nOutputs
        self.nPCpop = nPCpop
        self.nBasis = (self.nInputs*self.nIndvBasis) + self.nSharedBasis
        self.nPC = self.nPCpop*self.nOutputs
        self.range_delays = range_delays
        delays = np.linspace(self.range_delays[0], self.range_delays[1], self.nPCpop)              
        indxs_delays = ((self.range_delays[1] - delays)/self.dt).astype(int)
        self.indxs_delays = np.tile(indxs_delays, self.nOutputs)
        self.range_TC = range_TC
        self.range_scaling = range_scaling
        self.range_W = range_W

        # Purkinje cells' parameters and variables.
        self.PC_buffer = np.zeros((self.nPC, int(self.range_delays[1]/self.dt)))

        # Learning parameters and variables.
        self.beta_PF = beta_PF
        self.beta_MF = beta_MF

        self.create_synapses()
        self.create_basisFunctions()

    def create_synapses(self):

        # Mossy fiber synapses: specific granule processors for each individual input channel and one (plastic) granule processor shared for all Inputs.
        w_indvMF = np.zeros((self.nInputs, int(self.nInputs*self.nIndvBasis)))
        for i in np.arange(self.nInputs):
            w_indvMF[i, int(i*self.nIndvBasis):int((i+1)*self.nIndvBasis)] = 1

        w_sharedMF = np.random.uniform(self.range_W[0], self.range_W[1], self.nInputs*self.nSharedBasis).reshape((self.nInputs, self.nSharedBasis))
        self.w_MF = np.column_stack((w_indvMF, w_sharedMF))
        self.mask_wMF = np.column_stack((np.zeros((self.nInputs, int(self.nInputs*self.nIndvBasis))), np.ones((self.nInputs, self.nSharedBasis))))

        # Parallel fiber synpases: a population of heterogeneous delay-tuned Purkinje cells for each output channel (deep nucleus).
        self.w_PF = np.zeros(self.nBasis*self.nPC).reshape((self.nBasis, self.nPC)) 

    def create_basisFunctions(self):

        # Temporal basis parameters and variables.
        random_indxs = np.random.choice(np.arange(1000), self.nSharedBasis, replace=False)

        TC_indvReservoir = 1/np.logspace(np.log10(self.range_TC[0]), np.log10(self.range_TC[1]), self.nIndvBasis)
        TC_sharedReservoir = 1/np.logspace(np.log10(self.range_TC[0]), np.log10(self.range_TC[1]), 1000)
        t_constants = np.concatenate([np.tile(TC_indvReservoir, self.nInputs), TC_sharedReservoir[random_indxs]])
        self.gammas = np.exp(-self.dt*t_constants)

        scalingInput_indvReservoir = 1e-2/np.logspace(np.log10(self.range_scaling[0]), np.log10(self.range_scaling[1]), self.nIndvBasis)        
        scalingInput_sharedReservoir = 1e-2/np.logspace(np.log10(self.range_scaling[0]), np.log10(self.range_scaling[1]), 1000)
        self.scaling_input = np.concatenate([np.tile(scalingInput_indvReservoir, self.nInputs), scalingInput_sharedReservoir[random_indxs]])

        self.z = np.zeros(self.nBasis)
        self.p = np.zeros(self.nBasis)                                                          
        self.p_buffer = np.zeros((self.nBasis, int(self.range_delays[1]/self.dt)))

    # Activates the basis functions and computes the corresponding output, according to the actual weights.
    # Updates the weights for the basis, based on the modified Widrow-Hoff learning rule, or Least Mean Square method (LMS).
    def activate(self, input, error, update=True):

        # Compute output based on bases activity given the new input.
        x = np.dot(self.w_MF.T, input)                                       # Pons relay activity.
        self.z = self.z*self.gammas + self.scaling_input*x.flatten()
        self.p = self.p*self.gammas + self.z                                 # Granule cells activity based on alpha function.
        PC = np.dot(self.w_PF.T, self.p)                                     # Purkinje cells 'dendritic' activity.
        self.PC_buffer = np.column_stack((self.PC_buffer, PC))[:, 1:]        # FIFO-buffer for PC delayed output.
        delayed_PC = self.PC_buffer[np.arange(self.nPC), self.indxs_delays]  # Purkinje cell delayed output.
        delayed_PC = delayed_PC.reshape((self.nPCpop, self.nOutputs))
        DCN = np.sum(delayed_PC, axis=0)                                     # Deep Cerebellar Nucleus output as the sums of PC populations' delayed responses.

        if update:
            # Update PF synapses based on granule eligibility traces and climbing fiber signal (error).
            eligibility_traces = self.p_buffer[:, self.indxs_delays]
            CF = np.repeat(error.reshape((self.nOutputs, 1)), self.nBasis, axis=1).repeat(self.nPCpop, axis=0).T   # Climbing fiber teaching signal. (shape(error)[0]==self.nOutputs).
            self.w_PF += self.beta_PF * CF * eligibility_traces

            # Update MF synapses based on Oja's rule (stable Hebbian). Right now only the shared basis are updated ('mask_wMF').
            self.w_MF += self.beta_MF * (np.dot(input.reshape((input.shape[0],1)), [self.p]) - (self.p**2)*self.w_MF) * self.mask_wMF
            self.w_MF[self.w_MF<0] = 0

        # Update the buffer of basis' activity to be used for eligibility traces (proxy of dendritic activity in PC).
        self.p_buffer = np.column_stack((self.p_buffer, self.p))[:, 1:]                                            

        return DCN

    def test_basis(self, input_test, reset=True, save=False, file='basis.png'):

        if reset:
            self.reset()

        p = []
        for t in np.arange(input_test.shape[2]):
            self.activate(input=input_test[:,:,t][0], error=np.tile([0.], (self.nPC, 1)), update=False)
            p.append(self.p)
            
        p = np.array(p)
        
        plt.figure(figsize=(15,5))
        
        plt.plot(p)
        
        plt.yticks(fontsize=24)
        plt.ylabel('Activity (a.u.)', fontsize=26)
        plt.xlabel('Time (s)', fontsize=26)
        plt.xticks(np.linspace(0, input_test.shape[2], int(input_test.shape[2]*self.dt)+1), 
                   np.linspace(0, int(input_test.shape[2]*self.dt), int(input_test.shape[2]*self.dt)+1).astype(int), fontsize=24)
        
        sns.despine()
        plt.tight_layout()
        
        if save:
            plt.savefig(file, dpi=500)
            
        plt.show()

        if reset:
            self.reset()

    # Saves the weights of the Mossy Fibers (MF) and the Parallel Fibers (PF).
    def save_model(self, id=0, path=''):

        np.savetxt(path+'w_MF_'+str(id)+'.npy', self.w_MF)
        np.savetxt(path+'w_PF_'+str(id)+'.npy', self.w_PF)
        np.savetxt(path+'gammas_'+str(id)+'.npy', self.gammas)
        np.savetxt(path+'scaling_input_'+str(id)+'.npy', self.scaling_input)

    # Loads model: weights and time-related constants.
    def load_model(self, id=0, path=''):

        self.w_MF = np.loadtxt(path+'w_MF_'+str(id)+'.npy')
        self.w_PF = np.loadtxt(path+'w_PF_'+str(id)+'.npy')
        self.gammas = np.loadtxt(path+'gammas_'+str(id)+'.npy')
        self.scaling_input = np.loadtxt(path+'scaling_input_'+str(id)+'.npy')

    # Resets the whole module with the default values.
    def reset(self):

        self.__init__(dt=self.dt, nInputs=self.nInputs, nOutputs=self.nOutputs, nIndvBasis=self.nIndvBasis, nSharedBasis=self.nSharedBasis, 
                      beta_PF=self.beta_PF, beta_MF=self.beta_MF, range_delays=self.range_delays, range_TC=self.range_TC, 
                      range_scaling=self.range_scaling, range_W=self.range_W, nPCpop=self.nPCpop) 
      

# Plant coupled to a reactive controller.
class FeedbackLoop(object):
    
    # Creates an Inverted Pendulum controlled by a PD controller.
    def __init__(self, mass=67., height=0.85, dt=0.01, p_gain=1250., d_gain=250., delay=0.1):

        self.dt = dt                                                       # Simulation time step.
        self.delay = delay                                                 # Intrinsic delay of feedback's PD response to errors in angle.

        self.m = mass                                                      # Mass of the Inverted Pendulum (kilograms).
        self.h = height                                                    # Height from the center of mass (meters).
        self.g = 9.8                                                       # Gravity constant.
        self.th = 0.                                                       # Angle of the pendulum with respect to the vertical axis.
        self.th_buffer = np.zeros(int(self.delay/self.dt)).tolist()        # Prior history of angles=0. with length==delay.
        self.w = 0.                                                        # Angular velocity.
        self.previous_error = 0.                                           # Error in previous iteration.

        self.pGain = p_gain                                                # Proportional gain of the PD.
        self.dGain = d_gain                                                # Derivative gain of the PD.

    # The plant is modified according to the disturbance ('input_force'), the response of the feedback controller, and the input from the feedforward controller ('input_torque').
    def activate(self, input_PID, input_torque, input_force):
        
        error = -self.th_buffer[0] + input_PID                                          # Compute current error for the PD controller as the angle seconds ago (==delayed).
        output = self.pGain*error + self.dGain*((error-self.previous_error)/self.dt)    # Compute PD's output based on current angle, and the rate of change in angle.
        self.previous_error = error                                                     # Store current error for being used in next iteration as 'previous error'.

        # Computes instantaneous acceleration of non-linear plant with the inputs (see formula for the Inverted Pendulum).
        acc = (self.g/self.h)*np.sin(self.th) + (1./(self.m*self.h**2))*(output+input_torque) + (1./(self.m*self.h))*input_force*np.cos(self.th)
        self.w += acc * self.dt                                                         # Update angular velocity by Euler's method.
        self.th += self.w * self.dt                                                     # Update angle by Euler's method.

        self.th_buffer.append(self.th)                                                  # Store the angles to be passed with a delay to the PD.
        th_delayed = self.th_buffer.pop(0)

        return th_delayed, output                             # Return the angles without the delay, and the outputs of the PD controller, to be plotted at the end.
      
    # Resets the plant to its default values (angle=0, etc.).
    def reset(self):

        self.__init__(mass=self.m, height=self.h, dt=self.dt, p_gain=self.pGain, d_gain=self.dGain, delay=self.delay)



class CerebellarAgent(object):
    # Creates and agent composed of a feedback loop (inverted pendulum driven by a PD controller) and a feedforward module steering the feedback controller.
    def __init__(self, mass=67., height=0.85, dt=0.01, pd_gains=[1250., 250.], feedback_delay=0.1, nInputs=1, nOutputs=1, nPCpop=10, nIndvBasis=50, nSharedBasis=200, 
                 beta_MF=1e-3, beta_PF=1e-6, range_delays=[0.05, 0.5], range_TC=[0.05, 2.], range_scaling=[1, 100], range_W=[0., 1.]):

        self.type = type                     # HSPC, PSPC, or FEL.
        self.mass = mass                     # Mass of the Inverted Pendulum (in kilograms).
        self.height = height                 # Height of the pendulum (in meters).
        self.dt = dt                         # Time step of the simulation.
        self.range_delays = range_delays     # Delay (learning) of the feedforward module.
        self.feedback_delay = feedback_delay # Delay of the feedback loop.
        self.p_gain, self.d_gain = pd_gains  # Proportional and derivative gains of the PD controller (feedback loop).
        self.nInputs = nInputs
        self.nOutputs = nOutputs 
        self.nPCpop = nPCpop
        self.nIndvBasis = nIndvBasis
        self.nSharedBasis = nSharedBasis
        self.beta_MF = beta_MF
        self.beta_PF = beta_PF
        self.range_delays = range_delays
        self.range_TC = range_TC
        self.range_scaling = range_scaling
        self.range_W = range_W
        self.error = np.array([0.])

        self.ff = Cerebellum(dt=self.dt, nInputs=nInputs, nOutputs=nOutputs, nPCpop=nPCpop, nIndvBasis=nIndvBasis, nSharedBasis=nSharedBasis,
                             beta_MF=beta_MF, beta_PF=beta_PF, range_delays=range_delays, range_TC=range_TC, range_scaling=range_scaling, range_W=range_W)

        self.fb = FeedbackLoop(mass=self.mass, height=self.height, dt=self.dt, p_gain=self.p_gain, d_gain=self.d_gain, delay=self.feedback_delay)

    # Reset everything: feedback loop, feedforward module, and the agent itself.
    def reset(self):

        self.fb.reset()
        self.ff.reset()
        self.__init__(mass=self.mass, height=self.height, dt=self.dt, pd_gains=[self.p_gain, self.d_gain], nInputs=self.nInputs, nOutputs=self.nOutputs, nPCpop=self.nPCpop, 
                      nIndvBasis=self.nIndvBasis, nSharedBasis=self.nSharedBasis, beta_MF=self.beta_MF, beta_PF=self.beta_PF, range_delays=self.range_delays, range_TC=self.range_TC,
                      range_scaling=self.range_scaling, range_W=self.range_W, feedback_delay=self.feedback_delay)

    # The cue signal (x) activates the feedforward modules, and the outputs (along with the disturbance) are passed to the robot plant.
    def step(self, x, f, update=True):

        outputs = []
        ff_out = self.ff.activate(input=x, error=-self.error, update=update)
        outputs.append(ff_out)

        th_err_delayed, fb_out = self.fb.activate(input_PID=ff_out, input_torque=0., input_force=f)
        outputs.append(fb_out)
        self.error = np.array([th_err_delayed])

        th_err_delayed = th_err_delayed*(360./(2*np.pi))

        return th_err_delayed, outputs

    # Saves the ff modules.
    def save_agent(self, path=''):

        if path == '':
            path = self.type+'/'

        self.ff.save_model(id='000', path=path)

    # Loads the ff modules.
    def load_agent(self, path=''):

        if path == '':
            path = self.type+'/'

        self.ff.load_model(id='000', path=path)
