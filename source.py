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
    REF_MCS_INDICES = [0, 1, 3, 5, 8, 9, 11, 14, 16, 20, 22, 24, 25, 26, 27, 28]

    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler   = awgn_data['snr_vs_per']

    _, nrof_cqi = awgn_snr_vs_bler.shape

    bler = awgn_snr_vs_bler[:, REF_MCS_INDICES[ cqi ] ]

    if cqi == 0:
        return np.min(awgn_snr_range_dB)
    elif cqi == nrof_cqi - 1:
        return np.max(awgn_snr_range_dB)

    # Find the SNR indices closest to the REF_BLER_TARGET.
    # Estimate the instantaneous SNR by averaging these SNR values.
    # This assumes that the reported CQI actually had a BLER close to REF_BLER_TARGET.
    index1 = np.max(np.argwhere(REF_BLER_TARGET < bler))
    index2 = np.min(np.argwhere(REF_BLER_TARGET > bler))

    estimated_sinr_dB = (awgn_snr_range_dB[index1] + awgn_snr_range_dB[index2]) / 2.0

    return estimated_sinr_dB

def determine_cqi_from_sinr(snr_dB, packet_sizes, awgn_data, cqi_sinr_error = 0.0):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler   = awgn_data['snr_vs_per']

    REF_BLER_TARGET  = 0.1
    REF_MCS_INDICES = [0, 1, 3, 5, 8, 9, 11, 14, 16, 20, 22, 24, 25, 26, 27, 28]
    nrof_cqi = len( REF_MCS_INDICES )

    # Estimate the BLER for the reference MCSs used to calculate the CQI
    bler_at_snr = determine_bler_at_sinr(snr_dB + cqi_sinr_error, awgn_data)[ REF_MCS_INDICES ]
    
    # Calculate expcted throughput for all valid MCSs
    expected_tputs = np.multiply( ( 1 - bler_at_snr ), np.array( packet_sizes )[ REF_MCS_INDICES ] )
    
    # Ignore any MCSs with BLER less than REF_BLER_TARGET
    expected_tputs[ bler_at_snr > 0.1 ] = 0
    
    # The CQI is the index of the highest-throuput MCS from the reference MCSs
    cqi = 0
    if len( expected_tputs ) > 0:
        cqi = np.argmax( expected_tputs )
    
    return cqi
    

def determine_bler_at_sinr(snr_dB, awgn_data):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler   = awgn_data['snr_vs_per']

    _, nrof_mcs = awgn_snr_vs_bler.shape

    bler_at_sinr = np.ndarray((nrof_mcs))

    for i in range(nrof_mcs):
        bler = awgn_snr_vs_bler[:, i]
        
        if snr_dB <= np.min(awgn_snr_range_dB):
            bler_at_sinr[i] = 1.0
        elif snr_dB >= np.max(awgn_snr_range_dB):
            bler_at_sinr[i] = 0.0
        else:
            index1 = np.max(np.argwhere(awgn_snr_range_dB < snr_dB))
            index2 = np.min(np.argwhere(awgn_snr_range_dB > snr_dB))

            bler_at_sinr[i] = ( bler[index1] + bler[index2]) / 2.0

    return bler_at_sinr

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
    instantaneous_blers      = []
    channel_quality_indices = []
    for i in range( nrof_samples ):
        cqi_sinr_error = ( itpp.random.randn( ) - 0.5 ) * cqi_error_std
        
        snr_dB = 10 * np.log10( instantaneous_channel_snrs[i] )
        instantaneous_blers.append( determine_bler_at_sinr( snr_dB, awgn_data ) )
        channel_quality_indices.append( determine_cqi_from_sinr( snr_dB, packet_sizes, awgn_data, cqi_sinr_error) ) 
    
    return ( np.array( instantaneous_blers ), np.array( channel_quality_indices ) )
    

'''
'''
'''
Run simulation for the given set of parameters
'''
'''
'''
def run_simulation(pars, nrof_ttis):
    awgn_data = np.load( pars['awgn_datafile'], allow_pickle=True )[ ( ) ]

    snr_vs_bler = awgn_data['snr_vs_bler']
    snr_range_dB = awgn_data['snr_range_dB']

    nrof_snr, nrof_rates = snr_vs_bler.shape

    nrof_cqi = 16
        
    packet_sizes = [152, 200, 248, 320, 408, 504, 600, 712, 808, 936, 
                    936, 1032, 1192, 1352, 1544, 1736, 1800, 
                    1800, 1928, 2152, 2344, 2600, 2792, 2984, 3240, 3496, 3624, 3752, 4008]

    modorders    = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                    4, 4, 4, 4, 4, 4, 4, 
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    
    packet_error_probabilities, channel_quality_indices = simluate_rayleigh_fading_channel( nrof_ttis, 
                                                                                            pars['avg_snr_dB'], 
                                                                                            awgn_data, 
                                                                                            packet_sizes, 
                                                                                            pars['norm_doppler'], 
                                                                                            pars['seed'],
                                                                                            pars['cqi_error_std'])

    # Pre-generate ACK events for all rates for all channel samples
    packet_acks = np.ndarray( ( nrof_ttis, nrof_rates ) )
    for tti in range( nrof_ttis ):
        for rate_index in range( nrof_rates ):
            packet_acks[tti, rate_index] = np.random.uniform( ) > packet_error_probabilities[tti, rate_index]

            
    # Outer Loop Link Adaptation
    olla_bandit = OuterLoopLinkAdaptation(nrof_rates, packet_sizes, awgn_data, pars['target_bler'], pars['olla_step_size'])
    
    olla_rates  = []
    olla_acks  = []
    olla_tputs = []
    for tti in range( nrof_ttis ):
        
        # Skip the first few samples to account for CQI delay
        if tti < pars['cqi_delay']:
            selected_rate_index = np.random.randint(0, nrof_cqi)
            ack = packet_acks[tti, selected_rate_index]
        else:  
            cqi = channel_quality_indices[tti - pars['cqi_delay']]    
            selected_rate_index = olla_bandit.act( cqi )

            ack = packet_acks[tti, selected_rate_index]
            olla_bandit.update( selected_rate_index, cqi, ack )

        olla_rates.append(selected_rate_index)
        olla_acks.append(ack)
        olla_tputs.append( packet_sizes[ selected_rate_index ] * ack )
        
    
    # Thompson Sampling with Informed Priors
    bler_bler_cqi = np.ndarray( ( len( packet_sizes ), nrof_cqi ) )
    for cqi in range( nrof_cqi ):
        snr_dB = estimate_sinr_from_cqi(cqi, awgn_data)
        bler_bler_cqi[ :, cqi ] = determine_bler_at_sinr(snr_dB, awgn_data)

    
    ts_infp_bandit = ThompsonSamplingBandit(nrof_rates, packet_sizes, pars['target_bler'], bler_bler_cqi)
    
    ts_infp_rates  = []
    ts_infp_acks  = []
    ts_infp_tputs = []
    for tti in range( nrof_ttis ):
        
        # Skip the first few samples to account for CQI delay
        if tti < pars['cqi_delay']:
            selected_rate_index = np.random.randint(0, nrof_cqi)
            ack = packet_acks[tti, selected_rate_index]
        else:    
            cqi = channel_quality_indices[tti - pars['cqi_delay']]    
            selected_rate_index = ts_infp_bandit.act( cqi )

            ack = packet_acks[tti, selected_rate_index]
            ts_infp_bandit.update( selected_rate_index, cqi, ack )

        ts_infp_rates.append(selected_rate_index)
        ts_infp_acks.append(ack)
        ts_infp_tputs.append( packet_sizes[ selected_rate_index ] * ack )
    
    return ( olla_rates, olla_acks, olla_tputs, 
             ts_infp_rates, ts_infp_acks, ts_infp_tputs, )

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
                 target_bler):
        
        solvers.options['show_progress'] = False
        solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

        self.nrof_rates = nrof_rates
        self.packet_sizes = packet_sizes
        
        self.target_success_prob = 1.0 - target_bler
        
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
                 target_bler,
                 prior_bler=[],
                 prior_weight=100):
        
        super().__init__(nrof_rates, packet_sizes, target_bler)
        
        # Exploit prior knowledge
        if not prior_bler == []:  
            for cqi in range( prior_bler.shape[1] ):
                for rate_index in range(self.nrof_rates):                    
                    prior_mu = 1.0 - prior_bler[rate_index, cqi]
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
        
        expected_rewards = [( s * rew) for s, rew in zip(sampled_success_prob, self.packet_sizes)]

        return np.argmax(expected_rewards)

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
                 target_bler,
                 olla_step_size = 0.1):
        
        super().__init__(nrof_rates, packet_sizes, target_bler)
        
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

        bler_at_snr = determine_bler_at_sinr(adjusted_sinr, self.awgn_data)

        expected_rewards = [( (1.0 - bler) * rew) for bler, rew in zip( bler_at_snr, self.packet_sizes)]

        return np.argmax(expected_rewards)
