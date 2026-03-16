"""
Network Flow Feature Extractor for IoT Device Classification

Extracts flow-level features from pcap files to match the UNSW HomeNet dataset format.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scapy.all import rdpcap, IP, TCP, UDP
import warnings

warnings.filterwarnings('ignore')


class FlowFeatureExtractor:
    """Extract network flow features from pcap files."""
    
    def __init__(self):
        self.flows = defaultdict(lambda: {
            'packets': [],
            'fwd_packets': [],
            'bwd_packets': [],
            'start_time': None,
            'end_time': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'protocol': None
        })
    
    def _get_flow_key(self, pkt):
        """Generate a unique flow identifier."""
        if IP not in pkt:
            return None
        
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        protocol = pkt[IP].proto
        
        src_port = 0
        dst_port = 0
        
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        
        if (src_ip, src_port) < (dst_ip, dst_port):
            return (src_ip, src_port, dst_ip, dst_port, protocol)
        else:
            return (dst_ip, dst_port, src_ip, src_port, protocol)
    
    def _is_forward(self, pkt, flow_key):
        """Check if packet is in forward direction."""
        if IP not in pkt:
            return True
        
        src_ip = pkt[IP].src
        src_port = 0
        if TCP in pkt:
            src_port = pkt[TCP].sport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
        
        return (src_ip, src_port) == (flow_key[0], flow_key[1])
    
    def extract_from_pcap(self, pcap_path: str, max_packets: int = 100000) -> pd.DataFrame:
        """
        Extract flow features from a pcap file.
        
        Args:
            pcap_path: Path to pcap file
            max_packets: Maximum packets to process
            
        Returns:
            DataFrame with flow features
        """
        print(f"Reading pcap file: {pcap_path}")
        
        try:
            packets = rdpcap(pcap_path, count=max_packets)
        except Exception as e:
            raise ValueError(f"Error reading pcap file: {e}")
        
        print(f"Processing {len(packets)} packets...")
        
        self.flows = defaultdict(lambda: {
            'packets': [],
            'fwd_packets': [],
            'bwd_packets': [],
            'start_time': None,
            'end_time': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'protocol': None
        })
        
        for pkt in packets:
            if IP not in pkt:
                continue
            
            flow_key = self._get_flow_key(pkt)
            if flow_key is None:
                continue
            
            flow = self.flows[flow_key]
            pkt_time = float(pkt.time)
            pkt_len = len(pkt)
            
            if flow['start_time'] is None:
                flow['start_time'] = pkt_time
                flow['src_ip'] = flow_key[0]
                flow['src_port'] = flow_key[1]
                flow['dst_ip'] = flow_key[2]
                flow['dst_port'] = flow_key[3]
                flow['protocol'] = flow_key[4]
            
            flow['end_time'] = pkt_time
            flow['packets'].append({'time': pkt_time, 'len': pkt_len, 'pkt': pkt})
            
            if self._is_forward(pkt, flow_key):
                flow['fwd_packets'].append({'time': pkt_time, 'len': pkt_len})
            else:
                flow['bwd_packets'].append({'time': pkt_time, 'len': pkt_len})
        
        print(f"Extracted {len(self.flows)} flows")
        
        return self._compute_flow_features()
    
    def _safe_std(self, arr):
        """Compute std safely."""
        if len(arr) < 2:
            return 0.0
        return np.std(arr)
    
    def _safe_mean(self, arr):
        """Compute mean safely."""
        if len(arr) == 0:
            return 0.0
        return np.mean(arr)
    
    def _compute_iat(self, packets):
        """Compute inter-arrival times."""
        if len(packets) < 2:
            return []
        times = [p['time'] for p in packets]
        return [times[i+1] - times[i] for i in range(len(times)-1)]
    
    def _compute_flow_features(self) -> pd.DataFrame:
        """Compute features for all flows."""
        features_list = []
        
        for flow_key, flow in self.flows.items():
            if len(flow['packets']) < 2:
                continue
            
            fwd_lens = [p['len'] for p in flow['fwd_packets']]
            bwd_lens = [p['len'] for p in flow['bwd_packets']]
            all_lens = [p['len'] for p in flow['packets']]
            
            fwd_iat = self._compute_iat(flow['fwd_packets'])
            bwd_iat = self._compute_iat(flow['bwd_packets'])
            flow_iat = self._compute_iat(flow['packets'])
            
            duration = (flow['end_time'] - flow['start_time']) * 1e6
            
            total_fwd_bytes = sum(fwd_lens)
            total_bwd_bytes = sum(bwd_lens)
            total_bytes = total_fwd_bytes + total_bwd_bytes
            total_pkts = len(flow['packets'])
            
            features = {
                'SrcPort': flow['src_port'],
                'DstPort': flow['dst_port'],
                'Protocol': flow['protocol'],
                'FlowDuration': duration,
                'TotFwdPkts': len(fwd_lens),
                'TotBwdPkts': len(bwd_lens) if bwd_lens else 1,
                'TotLenFwdPkts': total_fwd_bytes,
                'TotLenBwdPkts': total_bwd_bytes,
                'FwdPktLenMax': max(fwd_lens) if fwd_lens else 0,
                'FwdPktLenMin': min(fwd_lens) if fwd_lens else 0,
                'FwdPktLenMean': self._safe_mean(fwd_lens),
                'FwdPktLenStd': self._safe_std(fwd_lens),
                'BwdPktLenMax': max(bwd_lens) if bwd_lens else 0,
                'BwdPktLenMin': min(bwd_lens) if bwd_lens else 0,
                'BwdPktLenMean': self._safe_mean(bwd_lens),
                'BwdPktLenStd': self._safe_std(bwd_lens),
                'FlowByts/s': total_bytes / (duration / 1e6) if duration > 0 else 0,
                'FlowPkts/s': total_pkts / (duration / 1e6) if duration > 0 else 0,
                'FlowIATMean': self._safe_mean(flow_iat) * 1e6 if flow_iat else 0,
                'FlowIATStd': self._safe_std(flow_iat) * 1e6 if flow_iat else 0,
                'FlowIATMax': max(flow_iat) * 1e6 if flow_iat else 0,
                'FlowIATMin': min(flow_iat) * 1e6 if flow_iat else 0,
                'FwdIATTot': sum(fwd_iat) * 1e6 if fwd_iat else 0,
                'FwdIATMean': self._safe_mean(fwd_iat) * 1e6 if fwd_iat else 0,
                'FwdIATStd': self._safe_std(fwd_iat) * 1e6 if fwd_iat else 0,
                'FwdIATMax': max(fwd_iat) * 1e6 if fwd_iat else 0,
                'FwdIATMin': min(fwd_iat) * 1e6 if fwd_iat else 0,
                'BwdIATTot': sum(bwd_iat) * 1e6 if bwd_iat else 0,
                'BwdIATMean': self._safe_mean(bwd_iat) * 1e6 if bwd_iat else 0,
                'BwdIATStd': self._safe_std(bwd_iat) * 1e6 if bwd_iat else 0,
                'BwdIATMax': max(bwd_iat) * 1e6 if bwd_iat else 0,
                'BwdIATMin': min(bwd_iat) * 1e6 if bwd_iat else 0,
                'FwdPSHFlags': 0,
                'BwdPSHFlags': 0,
                'FwdURGFlags': 0,
                'BwdURGFlags': 0,
                'FwdHeaderLen': len(fwd_lens) * 20,
                'BwdHeaderLen': len(bwd_lens) * 20 if bwd_lens else 0,
                'FwdPkts/s': len(fwd_lens) / (duration / 1e6) if duration > 0 else 0,
                'BwdPkts/s': len(bwd_lens) / (duration / 1e6) if duration > 0 else 0,
                'PktLenMin': min(all_lens) if all_lens else 0,
                'PktLenMax': max(all_lens) if all_lens else 0,
                'PktLenMean': self._safe_mean(all_lens),
                'PktLenStd': self._safe_std(all_lens),
                'PktLenVar': np.var(all_lens) if len(all_lens) > 1 else 0,
                'FINFlagCnt': 0,
                'SYNFlagCnt': 0,
                'RSTFlagCnt': 0,
                'PSHFlagCnt': 0,
                'ACKFlagCnt': 0,
                'URGFlagCnt': 0,
                'CWEFlagCount': 0,
                'ECEFlagCnt': 0,
                'Down/UpRatio': len(bwd_lens) / len(fwd_lens) if fwd_lens else 0,
                'PktSizeAvg': self._safe_mean(all_lens),
                'FwdSegSizeAvg': self._safe_mean(fwd_lens),
                'BwdSegSizeAvg': self._safe_mean(bwd_lens),
                'FwdByts/bAvg': 0,
                'FwdPkts/bAvg': 0,
                'FwdBlkRateAvg': 0,
                'BwdByts/bAvg': 0,
                'BwdPkts/bAvg': 0,
                'BwdBlkRateAvg': 0,
                'SubflowFwdPkts': len(fwd_lens),
                'SubflowFwdByts': total_fwd_bytes,
                'SubflowBwdPkts': len(bwd_lens) if bwd_lens else 1,
                'SubflowBwdByts': total_bwd_bytes,
                'InitFwdWinByts': -1,
                'InitBwdWinByts': -1,
                'FwdActDataPkts': len(fwd_lens),
                'FwdSegSizeMin': 0,
                'ActiveMean': 0,
                'ActiveStd': 0,
                'ActiveMax': 0,
                'ActiveMin': 0,
                '_src_ip': flow['src_ip'],
                '_dst_ip': flow['dst_ip'],
            }
            
            for pkt_info in flow['packets']:
                pkt = pkt_info['pkt']
                if TCP in pkt:
                    flags = pkt[TCP].flags
                    if flags & 0x01: features['FINFlagCnt'] = 1
                    if flags & 0x02: features['SYNFlagCnt'] = 1
                    if flags & 0x04: features['RSTFlagCnt'] = 1
                    if flags & 0x08: features['PSHFlagCnt'] = 1
                    if flags & 0x10: features['ACKFlagCnt'] = 1
                    if flags & 0x20: features['URGFlagCnt'] = 1
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)


def get_model_features():
    """Get the list of features expected by the model."""
    return [
        'SrcPort', 'DstPort', 'Protocol', 'FlowDuration', 'TotFwdPkts',
        'TotBwdPkts', 'TotLenFwdPkts', 'TotLenBwdPkts', 'FwdPktLenMax',
        'FwdPktLenMin', 'FwdPktLenMean', 'FwdPktLenStd', 'BwdPktLenMax',
        'BwdPktLenMin', 'BwdPktLenMean', 'BwdPktLenStd', 'FlowByts/s',
        'FlowPkts/s', 'FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FlowIATMin',
        'FwdIATTot', 'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin',
        'BwdIATTot', 'BwdIATMean', 'BwdIATStd', 'BwdIATMax', 'BwdIATMin',
        'FwdPSHFlags', 'BwdPSHFlags', 'FwdURGFlags', 'BwdURGFlags',
        'FwdHeaderLen', 'BwdHeaderLen', 'FwdPkts/s', 'BwdPkts/s', 'PktLenMin',
        'PktLenMax', 'PktLenMean', 'PktLenStd', 'PktLenVar', 'FINFlagCnt',
        'SYNFlagCnt', 'RSTFlagCnt', 'PSHFlagCnt', 'ACKFlagCnt', 'URGFlagCnt',
        'CWEFlagCount', 'ECEFlagCnt', 'Down/UpRatio', 'PktSizeAvg',
        'FwdSegSizeAvg', 'BwdSegSizeAvg', 'FwdByts/bAvg', 'FwdPkts/bAvg',
        'FwdBlkRateAvg', 'BwdByts/bAvg', 'BwdPkts/bAvg', 'BwdBlkRateAvg',
        'SubflowFwdPkts', 'SubflowFwdByts', 'SubflowBwdPkts', 'SubflowBwdByts',
        'InitFwdWinByts', 'InitBwdWinByts', 'FwdActDataPkts', 'FwdSegSizeMin',
        'ActiveMean', 'ActiveStd', 'ActiveMax', 'ActiveMin'
    ]
