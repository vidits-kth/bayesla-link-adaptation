import itpp
import numpy as np
from cvxopt import matrix, solvers


'''
'''
'''
CQI-related functions
'''
'''
'''
# Find the SINR for the given CQI to approximately achieve the given BLER target
def estimate_sinr_from_cqi(cqi, awgn_data):

    REF_BLER_TARGET = 0.1

    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_per   = awgn_data['snr_vs_bler']

    _, nrof_cqi = awgn_snr_vs_per.shape

    per = awgn_snr_vs_per[:, cqi]

    if cqi == 0:
        return np.min(awgn_snr_range_dB)
    elif cqi == nrof_cqi - 1:
        return np.max(awgn_snr_range_dB)

    index1 = np.max(np.argwhere(REF_BLER_TARGET < per))
    index2 = np.min(np.argwhere(REF_BLER_TARGET > per))

    estimated_sinr_dB = (awgn_snr_range_dB[index1] + awgn_snr_range_dB[index2]) / 2.0

    return estimated_sinr_dB

def determine_cqi_from_sinr(snr_dB, packet_sizes, awgn_data):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_per   = awgn_data['snr_vs_bler']

    _, nrof_cqi = awgn_snr_vs_per.shape
    REF_BLER_TARGET = 0.1

    per_at_snr = determine_per_at_sinr(snr_dB, awgn_data)

    # Find the largest MCS that has BLER less than the BLER target
    # The CQIs are evaluated in decreasing order and first value that predicts a BLER < 0.1
    # is returned.
    largest_mcs = 0
    for i in range( nrof_cqi ):
        current_mcs = nrof_cqi - i - 1
        if per_at_snr[current_mcs] < REF_BLER_TARGET:
            largest_mcs = current_mcs
            break
        else:
            continue

    # Determine the expected tput for all valid MCSs
    best_mcs = 0
    best_expected_tput = 0
    for i in range( largest_mcs ):
        expected_tput = ( 1 - per_at_snr[ i ] ) * float( packet_sizes[ i ] )
        if expected_tput > best_expected_tput:
            best_expected_tput = expected_tput
            best_mcs = i

    return best_mcs

def determine_per_at_sinr(snr_dB, awgn_data):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_per   = awgn_data['snr_vs_bler']

    _, nrof_cqi = awgn_snr_vs_per.shape

    per_at_sinr = np.ndarray((nrof_cqi))

    for i in range(nrof_cqi):
        per = awgn_snr_vs_per[:, i]
        if snr_dB <= np.min(awgn_snr_range_dB):
            per_at_sinr[i] = 1.0
        elif snr_dB >= np.max(awgn_snr_range_dB):
            per_at_sinr[i] = 0.0
        else:
            index1 = np.max(np.argwhere(awgn_snr_range_dB < snr_dB))
            index2 = np.min(np.argwhere(awgn_snr_range_dB > snr_dB))

            per_at_sinr[i] = (per[index1] + per[index2]) / 2.0

    return per_at_sinr

'''
'''
'''
Environment
'''
'''
'''
def simluate_rayleigh_fading_channel( nrof_samples, avg_snr_dB, awgn_data, packet_sizes, norm_doppler = 0.01, seed = 9999 ):
    
    # Create a Rayleigh fading channel. The channel power is normalized to 1 by default
    channel = itpp.comm.TDL_Channel( itpp.vec('0.0'), itpp.ivec('0') ) 
    channel.set_norm_doppler(norm_doppler)

    channel_coeff_itpp = itpp.cmat()
    channel.generate(nrof_samples, channel_coeff_itpp)

    channel_coeff = np.array( channel_coeff_itpp.get_col( 0 ) )
    
    avg_snr = 10 ** (0.1 * avg_snr_dB)
    instantaneous_channel_snrs = ( np.absolute( channel_coeff ) ** 2 ) * avg_snr
    
    _, nrof_rates = awgn_data['snr_vs_bler'].shape
    instantaneous_pers      = []
    channel_quality_indices = []
    for i in range( nrof_samples ):
        snr_dB = 10 * np.log10( instantaneous_channel_snrs[i] )
        instantaneous_pers.append( determine_per_at_sinr( snr_dB, awgn_data ) )
        channel_quality_indices.append( determine_cqi_from_sinr( snr_dB, packet_sizes, awgn_data) ) 
    
    return ( np.array( instantaneous_pers ), np.array( channel_quality_indices ) )
    

'''
'''
'''
Base Constrained Bandit 
'''
'''
'''
# nrof_rates: Number of bandit arms (K)
# packet_sizes: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability
# window_size: Window size for sliding window bandit. Events outside the window are discarded
class BaseConstrainedBandit():
    def __init__(self, 
                 nrof_rates, 
                 packet_sizes, 
                 target_per):
        
        solvers.options['show_progress'] = False
        solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

        self.nrof_rates = nrof_rates
        self.packet_sizes = packet_sizes
        
        self.target_success_prob = 1.0 - target_per
        
        nrof_cqis = nrof_rates
        
        self.ack_count  = np.zeros( ( nrof_rates, nrof_cqis ) )
        self.nack_count = np.zeros( ( nrof_rates, nrof_cqis ) )
       
    # Determine which arm to be pulled
    def act(self, cqi): # Implemented in child classes
        pass
    
    # Update the bandit
    def update(self, rate_index, cqi, ack):  
        #self.t += 1
        
        if ack:
            self.ack_count[ rate_index, cqi ] += 1
        else:
            self.nack_count[ rate_index, cqi ] += 1
         
    # Calculate the selection probability vector by solving a linear program
    def calculate_selection_probabilities(self, success_prob, tolerance=1e-5):

        c = matrix(-1 * np.array(success_prob) * np.array(self.packet_sizes))

        neg_success_prob = [-1.0 * r for r in success_prob]
        
        G = matrix(np.vstack([neg_success_prob, -1.0 * np.eye(self.nrof_rates)]))
        h = matrix(np.append(-1 * self.target_success_prob, np.zeros((1, self.nrof_rates))))

        A = matrix(np.ones((1, self.nrof_rates)))
        b = matrix([1.0])

        sol = solvers.lp(c, G, h, A, b)
        
        selection_prob = np.reshape(np.array(sol['x']), -1)
        
        if None in selection_prob: # Unsolvable optimiation
            return [None]
            
        # Fix numerical issues
        selection_prob[np.abs(selection_prob) < tolerance] = 0.0  # Remove precision-related values
        selection_prob = selection_prob / sum(selection_prob)     # Recalibrate probability vector to sum to 1
        
        return selection_prob
    
    # Sample from the probabilistic selection vector
    def sample_prob_selection_vector(self, prob):
        try:
            return np.argwhere(np.random.multinomial(1, prob))[0][0]
            # return dependent_rounding(prob)
        except: # Throws ValueError somtimes
            print('Error thrown by prob sampling. Returning random sample')
            return np.random.randint(0, self.nrof_rates)
        
'''
'''
'''
Oracle Constrained Bandit
'''
'''
'''
# nrof_rates: Number of bandit arms (K)
# packet_sizes: Reward value for each arm (r_k) if successful
# target_success_prob: Target success probability
# success_prob: Success probability for each bandit arm
class OracleConstrainedBandit(BaseConstrainedBandit):
    def __init__(self, 
                 nrof_rates, 
                 packet_sizes, 
                 target_per,
                 env_instance=None):
        
        super().__init__(nrof_rates, packet_sizes, target_per)
        self.env = env_instance
    
    # Determine which arm to be pulled
    def act(self):
        success_prob = self.env.get_success_prob(self.t)
        selection_prob = self.calculate_selection_probabilities(success_prob)
        
       # print(success_prob)
        
        return self.sample_prob_selection_vector(selection_prob)
    
    # Get selection probabilties (for debugging purposes)
    def get_selection_prob(self):
        return self.prob

'''
'''
'''
Thompson Sampling Bandit
Provides:
 (i) Unimodal Thompson sampling (UTS)
 (ii) Constrained Thompson sampling (Con-TS)
'''
'''
'''
    
class ThompsonSamplingBandit(BaseConstrainedBandit):
    def __init__(self, 
                 nrof_rates, 
                 packet_sizes, 
                 target_per,
                 prior_success_mean=[]):
        
        super().__init__(nrof_rates, packet_sizes, target_per)
        
        # Exploit prior knowledge
        if not prior_success_mean == []:
            self.informed_prior = True
            
            prior_weight = 10
            for arm in range(self.nrof_rates):
                self.success_count[arm] = 1 + int(prior_weight * prior_success_mean[arm])
                
                fail_count = 1 + int(prior_weight * (1.0 - prior_success_mean[arm]))
                self.pull_count[arm] = self.success_count[arm] + fail_count
        else:
            self.informed_prior = False
                
    # Determine which arm to be pulled
    def act(self, cqi):
        # Ensure that each arm is pulled at least once
#        if not self.informed_prior:
#            if self.t < self.nrof_rates:
#                return self.t
        
        # Sample a success probability from beta distribution Beta(a, b)
        # where a = 1 + self.ack_count[ cqi, rate_index ]
        # and   b = 1 + self.nack_count[ cqi, rate_index ]
        sampled_success_prob = [ np.random.beta(1 + self.ack_count[ rate_index, cqi  ], 
                                                1 + self.nack_count[ rate_index, cqi ] ) 
                                for rate_index in range(self.nrof_rates)]
        
        #sampled_expected_rewards = [(suc * rew) for suc, rew in zip(sampled_success_prob, self.packet_sizes)]
        #return np.argmax(sampled_expected_rewards)
            
        #if self.t % 1000 == 0:
        #    print(sampled_reward_event_prob)
        #    print([x + y for x, y in zip(self.reward_event_count, self.no_reward_event_count)])
        
        # Success probability constraint through linear programming
        selection_probabilities = self.calculate_selection_probabilities(sampled_success_prob)
        if None in selection_probabilities: # Unsolvable optimization
            #if self.t % 1000 == 0:
            #    print('No solution found!')
                
            return np.random.randint(0, self.nrof_rates)
            #sampled_expected_rewards = [(suc * rew) for suc, rew in zip(sampled_success_prob, self.packet_sizes)]
            #return np.argmax(sampled_expected_rewards)
        else:
            return self.sample_prob_selection_vector( selection_probabilities )   
        

'''
'''
'''
Outer Loop Link Adaptation: Bandit-like interface for OLLA
'''
'''
'''

class OuterLoopLinkAdaptation(BaseConstrainedBandit):
    def __init__(self, 
                 nrof_rates, 
                 packet_sizes, 
                 awgn_data,
                 target_per,
                 olla_step_size = 0.1):
        
        super().__init__(nrof_rates, packet_sizes, target_per)
        
        self.awgn_data = awgn_data

        self.sinr_offset = 0.0
        self.olla_step_size = 0.1

    def update(self, rate_index, cqi, ack):
        if ack:
            self.sinr_offset +=  self.olla_step_size
        else:
            self.sinr_offset -= self.target_success_prob / (1.0 - self.target_success_prob) * self.olla_step_size 
        
    def act(self, cqi):

        if cqi == 0:
            return 0
        else:
            estimated_sinr = estimate_sinr_from_cqi(cqi, self.awgn_data )
            adjusted_sinr = estimated_sinr + self.sinr_offset

            per_at_snr = determine_per_at_sinr(adjusted_sinr, self.awgn_data)
            
            expected_rewards = [( (1.0 - per) * rew) for per, rew in zip(per_at_snr, self.packet_sizes)]

            return np.argmax(expected_rewards)
