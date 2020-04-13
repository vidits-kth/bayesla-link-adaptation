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
# Find the SINR for the given CQI to approximately achieve the given PER target
def estimate_sinr_from_cqi(cqi, awgn_data):

    REF_PER_TARGET = 0.1
    REF_MCS_INDICES = [0, 1, 3, 5, 8, 9, 11, 14, 16, 20, 22, 24, 25, 26, 27, 28]

    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_per   = awgn_data['snr_vs_per']

    _, nrof_cqi = awgn_snr_vs_per.shape

    per = awgn_snr_vs_per[:, REF_MCS_INDICES[ cqi ] ]

    if cqi == 0:
        return np.min(awgn_snr_range_dB)
    elif cqi == nrof_cqi - 1:
        return np.max(awgn_snr_range_dB)

    # Find the SNR indices closest to the REF_PER_TARGET.
    # Estimate the instantaneous SNR by averaging these SNR values.
    # This assumes that the reported CQI actually had a PER close to REF_PER_TARGET.
    index1 = np.max(np.argwhere(REF_PER_TARGET < per))
    index2 = np.min(np.argwhere(REF_PER_TARGET > per))

    estimated_sinr_dB = (awgn_snr_range_dB[index1] + awgn_snr_range_dB[index2]) / 2.0

    return estimated_sinr_dB

def determine_cqi_from_sinr(snr_dB, packet_sizes, awgn_data, cqi_sinr_error = 0.0):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_per   = awgn_data['snr_vs_per']

    REF_PER_TARGET  = 0.1
    REF_MCS_INDICES = [0, 1, 3, 5, 8, 9, 11, 14, 16, 20, 22, 24, 25, 26, 27, 28]
    nrof_cqi = len( REF_MCS_INDICES )

    # Estimate the PER for the reference MCSs used to calculate the CQI
    per_at_snr = determine_per_at_sinr(snr_dB + cqi_sinr_error, awgn_data)[ REF_MCS_INDICES ]
    
    # Calculate expcted throughput for all valid MCSs
    expected_tputs = np.multiply( ( 1 - per_at_snr ), np.array( packet_sizes )[ REF_MCS_INDICES ] )
    
    # Ignore any MCSs with PER less than REF_PER_TARGET
    expected_tputs[ per_at_snr > 0.1 ] = 0
    
    # The CQI is the index of the highest-throuput MCS from the reference MCSs
    cqi = 0
    if len( expected_tputs ) > 0:
        cqi = np.argmax( expected_tputs )
    
    return cqi
    

def determine_per_at_sinr(snr_dB, awgn_data):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_per   = awgn_data['snr_vs_per']

    _, nrof_mcs = awgn_snr_vs_per.shape

    per_at_sinr = np.ndarray((nrof_mcs))

    for i in range(nrof_mcs):
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
def simluate_rayleigh_fading_channel( nrof_samples, avg_snr_dB, awgn_data, packet_sizes, norm_doppler = 0.01, seed = 9999, cqi_error_std = 0.0 ):
    
    # Create a Rayleigh fading channel. The channel power is normalized to 1 by default
    channel = itpp.comm.TDL_Channel( itpp.vec('0.0'), itpp.ivec('0') ) 
    channel.set_norm_doppler(norm_doppler)

    channel_coeff_itpp = itpp.cmat()
    channel.generate(nrof_samples, channel_coeff_itpp)

    channel_coeff = np.array( channel_coeff_itpp.get_col( 0 ) )
    
    avg_snr = 10 ** (0.1 * avg_snr_dB)
    instantaneous_channel_snrs = ( np.absolute( channel_coeff ) ** 2 ) * avg_snr
    
    _, nrof_rates = awgn_data['snr_vs_per'].shape
    instantaneous_pers      = []
    channel_quality_indices = []
    for i in range( nrof_samples ):
        cqi_sinr_error = ( itpp.random.randn( ) - 0.5 ) * cqi_error_std
        
        snr_dB = 10 * np.log10( instantaneous_channel_snrs[i] )
        instantaneous_pers.append( determine_per_at_sinr( snr_dB, awgn_data ) )
        channel_quality_indices.append( determine_cqi_from_sinr( snr_dB, packet_sizes, awgn_data, cqi_sinr_error) ) 
    
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
                 prior_per=[],
                 prior_weight=100):
        
        super().__init__(nrof_rates, packet_sizes, target_per)
        
        # Exploit prior knowledge
        if not prior_per == []:  
            for cqi in range( prior_per.shape[1] ):
                for rate_index in range(self.nrof_rates):                    
                    prior_mu = 1.0 - prior_per[rate_index, cqi]
                    self.ack_count[rate_index, cqi] = int( prior_weight * ( prior_mu  ) )
                    self.nack_count[rate_index, cqi] = int( prior_weight * ( 1.0 - prior_mu ) )
                
    # Determine which arm to be pulled
    def act(self, cqi):
        
        # Sample a success probability from beta distribution Beta(a, b)
        # where a = 1 + self.ack_count[ cqi, rate_index ]
        # and   b = 1 + self.nack_count[ cqi, rate_index ]
        sampled_success_prob = [ np.random.beta(1 + self.ack_count[ rate_index, cqi  ], 
                                                1 + self.nack_count[ rate_index, cqi ] ) 
                                for rate_index in range(self.nrof_rates)]
        
        # Success probability constraint through linear programming
        selection_probabilities = self.calculate_selection_probabilities(sampled_success_prob)
        if None in selection_probabilities: # Unsolvable optimization
            return np.random.randint(0, self.nrof_rates)
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
        self.olla_step_size = olla_step_size

    def update(self, rate_index, cqi, ack):
        if ack:
            self.sinr_offset +=  self.olla_step_size
        else:
            self.sinr_offset -= self.target_success_prob / (1.0 - self.target_success_prob) * self.olla_step_size 
        
    def act(self, cqi):

        estimated_sinr = estimate_sinr_from_cqi(cqi, self.awgn_data )
        adjusted_sinr = estimated_sinr + self.sinr_offset

        per_at_snr = determine_per_at_sinr(adjusted_sinr, self.awgn_data)

        expected_rewards = [( (1.0 - per) * rew) for per, rew in zip(per_at_snr, self.packet_sizes)]

        return np.argmax(expected_rewards)
