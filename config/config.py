import os
from dotenv import load_dotenv

load_dotenv()  # TODO: replace load_dotenv logic with pydantic

class Config:

    rand_state = 42
    COLUMN_LABEL = 'label'
    COLUMN_LABEL_CAT = 'label_cat'
    COLUMN_LABEL_IS_ATTACK = 'label_is_attack'
    LABEL_BENIGN = 'Benign'
    LABEL_CAT_MAPPING = {
        'Benign': 0,
        'Bot': 1,
        'DoS attacks-GoldenEye': 4,
        'DoS attacks-Hulk': 5,
        'DoS attacks-SlowHTTPTest': 6,
        'DoS attacks-Slowloris': 7,
        'DDOS attack-HOIC': 2,
        'DDoS attacks-LOIC-HTTP': 3,
        'FTP-BruteForce': 8,
        'Infilteration': 9,
        'SSH-Bruteforce': 10,
    }

    # paths
    DATASETS_FOLDER = os.getenv('DATASETS_FOLDER')
    STATS_AND_IMAGES_FOLDER = os.getenv('STATS_AND_IMAGES_FOLDER')
    MODELS_FOLDER = os.getenv('MODELS_FOLDER')
    FEDERATED_FOLDER = os.getenv('FEDERATED_FOLDER')
    CIC_IDS_2018_PROCESSED_CSVS = os.getenv('CIC_IDS_2018_PROCESSED_CSVS')    

    data_types = {
        'dst_port': 'uint32',
        'protocol': 'uint8',
        'timestamp': 'object',
        'flow_duration': 'int64',
        'tot_fwd_pkts': 'uint32',
        'tot_bwd_pkts': 'uint32',
        'totlen_fwd_pkts': 'uint32',
        'totlen_bwd_pkts': 'uint32',
        'fwd_pkt_len_max': 'uint16',
        'fwd_pkt_len_min': 'uint16',
        'fwd_pkt_len_mean': 'float32',
        'fwd_pkt_len_std': 'float32',
        'bwd_pkt_len_max': 'uint16',
        'bwd_pkt_len_min': 'uint16',
        'bwd_pkt_len_mean': 'float32',
        'bwd_pkt_len_std': 'float32',
        'flow_byts_s': 'float64',
        'flow_pkts_s': 'float64',
        'flow_iat_mean': 'float32',
        'flow_iat_std': 'float32',
        'flow_iat_max': 'int64',
        'flow_iat_min': 'int64',
        'fwd_iat_tot': 'int64',
        'fwd_iat_mean': 'float32',
        'fwd_iat_std': 'float32',
        'fwd_iat_max': 'int64',
        'fwd_iat_min': 'int64',
        'bwd_iat_tot': 'uint32',
        'bwd_iat_mean': 'float32',
        'bwd_iat_std': 'float32',
        'bwd_iat_max': 'uint32',
        'bwd_iat_min': 'uint32',
        'fwd_psh_flags': 'uint8',
        'bwd_psh_flags': 'uint8',
        'fwd_urg_flags': 'uint8',
        'bwd_urg_flags': 'uint8',
        'fwd_header_len': 'uint32',
        'bwd_header_len': 'uint32',
        'fwd_pkts_s': 'float32',
        'bwd_pkts_s': 'float32',
        'pkt_len_min': 'uint16',
        'pkt_len_max': 'uint16',
        'pkt_len_mean': 'float32',
        'pkt_len_std': 'float32',
        'pkt_len_var': 'float32',
        'fin_flag_cnt': 'uint8',
        'syn_flag_cnt': 'uint8',
        'rst_flag_cnt': 'uint8',
        'psh_flag_cnt': 'uint8',
        'ack_flag_cnt': 'uint8',
        'urg_flag_cnt': 'uint8',
        'cwe_flag_count': 'uint8',
        'ece_flag_cnt': 'uint8',
        'down_up_ratio': 'uint16',
        'pkt_size_avg': 'float32',
        'fwd_seg_size_avg': 'float32',
        'bwd_seg_size_avg': 'float32',
        'fwd_byts_b_avg': 'uint8',
        'fwd_pkts_b_avg': 'uint8',
        'fwd_blk_rate_avg': 'uint8',
        'bwd_byts_b_avg': 'uint8',
        'bwd_pkts_b_avg': 'uint8',
        'bwd_blk_rate_avg': 'uint8',
        'subflow_fwd_pkts': 'uint32',
        'subflow_fwd_byts': 'uint32',
        'subflow_bwd_pkts': 'uint32',
        'subflow_bwd_byts': 'uint32',
        'init_fwd_win_byts': 'int32',
        'init_bwd_win_byts': 'int32',
        'fwd_act_data_pkts': 'uint32',
        'fwd_seg_size_min': 'uint8',
        'active_mean': 'float32',
        'active_std': 'float32',
        'active_max': 'uint32',
        'active_min': 'uint32',
        'idle_mean': 'float32',
        'idle_std': 'float32',
        'idle_max': 'uint64',
        'idle_min': 'uint64',
        'label': 'category'
    }

    FEATURES_NO_VARIANCE = [
        "bwd_blk_rate_avg",
        "bwd_byts_b_avg",
        "bwd_pkts_b_avg",
        "bwd_psh_flags",
        "bwd_urg_flags",
        "fwd_blk_rate_avg",
        "fwd_byts_b_avg",
        "fwd_pkts_b_avg"
    ]