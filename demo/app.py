"""
IoT Device Classification Web Demo

Upload a pcap file and classify IoT devices based on network traffic patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.feature_extractor import FlowFeatureExtractor, get_model_features

st.set_page_config(
    page_title="IoT Device Classifier",
    page_icon="🔌",
    layout="wide"
)

DEVICE_ICONS = {
    'Audio': '🔊',
    'Camera': '📷',
    'Hub': '🏠',
    'Lighting': '💡',
    'Motion_Sensor': '🚶',
    'PC': '💻',
    'PowerOutlet': '🔌',
    'Scale': '⚖️',
    'baby_monitor': '👶',
    'power_switch': '🔘',
    'printer': '🖨️',
    'router': '📡',
    'sleep_sensor': '😴',
    'smartphone': '📱'
}

DEVICE_DESCRIPTIONS = {
    'Audio': 'Smart speakers, voice assistants (e.g., Amazon Echo, Google Home)',
    'Camera': 'IP cameras, security cameras, webcams, smart doorbells',
    'Hub': 'Smart home hubs, IoT gateways, home automation controllers',
    'Lighting': 'Smart bulbs, LED strips, connected light switches',
    'Motion_Sensor': 'Motion detectors, presence sensors, PIR sensors',
    'PC': 'Desktop computers, laptops, workstations',
    'PowerOutlet': 'Smart power outlets, energy monitoring plugs',
    'Scale': 'Smart scales, health monitors, body composition analyzers',
    'baby_monitor': 'Baby monitors, nursery cameras, audio monitors',
    'power_switch': 'Smart switches, Wemo switches, relay controllers',
    'printer': 'Network printers, wireless printers, print servers',
    'router': 'Network routers, gateways, access points',
    'sleep_sensor': 'Sleep tracking devices, bed sensors, sleep monitors',
    'smartphone': 'Mobile phones, tablets, portable devices'
}

def init_session_state():
    """Initialize session state variables safely."""
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = None


def get_current_page():
    """Get current page safely."""
    return st.session_state.get('page', 'upload')


def navigate_to(page, device=None):
    """Navigate to a different page."""
    st.session_state.page = page
    if device:
        st.session_state.selected_device = device


@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, 'models', 'xgboost.joblib')
    scaler_path = os.path.join(base_path, 'data', 'processed', 'scaler.joblib')
    metadata_path = os.path.join(base_path, 'data', 'processed', 'metadata.joblib')
    
    if not os.path.exists(model_path):
        return None, None, None
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)
    
    return model, scaler, metadata


def predict_devices(df, model, scaler, metadata):
    """Make predictions on the extracted features."""
    feature_cols = metadata['feature_names']
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    label_mapping_inv = {v: k for k, v in metadata['label_mapping'].items()}
    predicted_labels = [label_mapping_inv[p] for p in predictions]
    
    return predicted_labels, probabilities, list(metadata['label_mapping'].keys())


def render_sidebar():
    """Render the sidebar."""
    st.sidebar.title("🔌 IoT Classifier")
    st.sidebar.markdown("---")
    
    st.sidebar.info(
        """
        **Model:** XGBoost  
        **Accuracy:** 88.78%  
        **Dataset:** UNSW HomeNet  
        **Device Types:** 14
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Supported Devices")
    for device, icon in DEVICE_ICONS.items():
        st.sidebar.markdown(f"{icon} {device}")


def render_upload_page():
    """Render the upload page."""
    st.title("🔌 IoT Device Classifier")
    st.markdown("#### Classify IoT devices from network traffic (PCAP files)")
    
    model, scaler, metadata = load_model()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please train the model first.")
        return
    
    st.success("✅ Model loaded successfully!")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload PCAP File")
        uploaded_file = st.file_uploader(
            "Choose a PCAP file",
            type=['pcap', 'pcapng', 'cap'],
            help="Upload a network capture file to analyze"
        )
        
        max_packets = st.slider(
            "Maximum packets to analyze",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )
    
    with col2:
        st.subheader("ℹ️ How it works")
        st.markdown("""
        1. 📤 Upload a PCAP file
        2. 🔍 System extracts network flows
        3. ⚙️ Features are computed
        4. 🤖 ML model classifies devices
        5. 📊 View results & details
        """)
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("🔄 Analyzing network traffic..."):
                extractor = FlowFeatureExtractor()
                df = extractor.extract_from_pcap(tmp_path, max_packets=max_packets)
            
            if len(df) == 0:
                st.warning("⚠️ No valid network flows found in the PCAP file.")
                return
            
            st.success(f"✅ Extracted **{len(df)}** network flows")
            
            with st.spinner("🤖 Classifying devices..."):
                predictions, probabilities, class_names = predict_devices(
                    df, model, scaler, metadata
                )
            
            df['Predicted_Device'] = predictions
            df['Confidence'] = [max(prob) * 100 for prob in probabilities]
            
            st.session_state.results_df = df
            st.session_state.page = 'results'
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        st.markdown("---")
        st.subheader("🎮 Don't have a PCAP file?")
        st.markdown("""
        Capture network traffic using:
        
        ```bash
        # macOS/Linux
        sudo tcpdump -i en0 -w capture.pcap -c 5000
        
        # Windows - Use Wireshark
        ```
        
        Or download samples from [Wireshark](https://wiki.wireshark.org/SampleCaptures)
        """)


def render_results_page():
    """Render the results overview page."""
    df = st.session_state.results_df
    
    if df is None:
        st.session_state.page = 'upload'
        st.rerun()
        return
    
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to Upload", use_container_width=True):
            st.session_state.results_df = None
            st.session_state.page = 'upload'
            st.rerun()
    
    with col_title:
        st.title("📊 Classification Results")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    device_counts = df['Predicted_Device'].value_counts()
    
    with col1:
        st.metric("📦 Total Flows", len(df))
    with col2:
        st.metric("🔌 Unique Devices", len(device_counts))
    with col3:
        st.metric("🎯 Avg Confidence", f"{df['Confidence'].mean():.1f}%")
    with col4:
        top_device = device_counts.index[0]
        st.metric("🏆 Most Common", f"{DEVICE_ICONS.get(top_device, '')} {top_device}")
    
    st.markdown("---")
    st.subheader("🎯 Detected Devices (Click for Details)")
    
    num_cols = min(4, len(device_counts))
    cols = st.columns(num_cols)
    
    for idx, (device, count) in enumerate(device_counts.items()):
        with cols[idx % num_cols]:
            icon = DEVICE_ICONS.get(device, '❓')
            avg_conf = df[df['Predicted_Device'] == device]['Confidence'].mean()
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
                <h1 style="margin: 0; font-size: 3em;">{icon}</h1>
                <h4 style="margin: 5px 0;">{device}</h4>
                <p style="margin: 0;"><b>{count}</b> flows</p>
                <p style="margin: 0; color: gray; font-size: 0.9em;">Confidence: {avg_conf:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"View Details", key=f"btn_{device}", use_container_width=True):
                st.session_state.selected_device = device
                st.session_state.page = 'device_detail'
                st.rerun()
    
    st.markdown("---")
    st.subheader("📈 Device Distribution")
    
    fig_pie = px.pie(
        values=device_counts.values,
        names=device_counts.index,
        title="",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(showlegend=True, height=400)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    col_detail, col_download = st.columns([1, 1])
    
    with col_detail:
        if st.button("🔍 View All Flow Details", use_container_width=True):
            st.session_state.selected_device = None
            st.session_state.page = 'all_details'
            st.rerun()
    
    with col_download:
        display_df = df[['_src_ip', '_dst_ip', 'SrcPort', 'DstPort', 'Protocol', 
                        'Predicted_Device', 'Confidence']].copy()
        display_df.columns = ['Source IP', 'Dest IP', 'Src Port', 'Dst Port', 
                             'Protocol', 'Device Type', 'Confidence (%)']
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="iot_classification_results.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_device_detail_page():
    """Render the device detail page."""
    df = st.session_state.results_df
    device = st.session_state.selected_device
    
    if df is None or device is None:
        st.session_state.page = 'results'
        st.rerun()
        return
    
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
    
    icon = DEVICE_ICONS.get(device, '❓')
    with col_title:
        st.title(f"{icon} {device} - Details")
    
    st.markdown("---")
    
    device_df = df[df['Predicted_Device'] == device].copy()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Flows", len(device_df))
    with col2:
        st.metric("Avg Confidence", f"{device_df['Confidence'].mean():.1f}%")
    with col3:
        st.metric("Min Confidence", f"{device_df['Confidence'].min():.1f}%")
    
    st.markdown("---")
    st.subheader("📝 Device Description")
    st.info(DEVICE_DESCRIPTIONS.get(device, "IoT device detected in network traffic."))
    
    st.markdown("---")
    st.subheader("📊 Confidence Distribution")
    
    fig_hist = px.histogram(
        device_df, 
        x='Confidence',
        nbins=20,
        title=f"Confidence Score Distribution for {device}",
        color_discrete_sequence=['#4CAF50']
    )
    fig_hist.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title="Number of Flows",
        height=300
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Flow Details")
    
    display_df = device_df[['_src_ip', '_dst_ip', 'SrcPort', 'DstPort', 'Protocol', 'Confidence']].copy()
    display_df.columns = ['Source IP', 'Dest IP', 'Src Port', 'Dst Port', 'Protocol', 'Confidence (%)']
    display_df['Confidence (%)'] = display_df['Confidence (%)'].round(1)
    
    st.dataframe(
        display_df.style.background_gradient(subset=['Confidence (%)'], cmap='Greens'),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Results Overview", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
    with col2:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {device} Data",
            data=csv,
            file_name=f"iot_{device}_flows.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_all_details_page():
    """Render all flow details page."""
    df = st.session_state.results_df
    
    if df is None:
        st.session_state.page = 'results'
        st.rerun()
        return
    
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
    
    with col_title:
        st.title("🔍 All Flow Details")
    
    st.markdown("---")
    
    device_filter = st.multiselect(
        "Filter by Device Type",
        options=df['Predicted_Device'].unique().tolist(),
        default=df['Predicted_Device'].unique().tolist()
    )
    
    confidence_range = st.slider(
        "Confidence Range (%)",
        min_value=0,
        max_value=100,
        value=(0, 100)
    )
    
    filtered_df = df[
        (df['Predicted_Device'].isin(device_filter)) &
        (df['Confidence'] >= confidence_range[0]) &
        (df['Confidence'] <= confidence_range[1])
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} flows**")
    
    st.markdown("---")
    
    display_df = filtered_df[['_src_ip', '_dst_ip', 'SrcPort', 'DstPort', 'Protocol', 
                              'Predicted_Device', 'Confidence']].copy()
    display_df.columns = ['Source IP', 'Dest IP', 'Src Port', 'Dst Port', 
                         'Protocol', 'Device Type', 'Confidence (%)']
    display_df['Confidence (%)'] = display_df['Confidence (%)'].round(1)
    
    st.dataframe(
        display_df.style.background_gradient(subset=['Confidence (%)'], cmap='Greens'),
        use_container_width=True,
        height=500
    )
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Results Overview", use_container_width=True, key="back_btn_2"):
            st.session_state.page = 'results'
            st.rerun()
    with col2:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="iot_filtered_flows.csv",
            mime="text/csv",
            use_container_width=True
        )


def main():
    """Main application entry point."""
    # Initialize session state first
    init_session_state()
    
    render_sidebar()
    
    current_page = get_current_page()
    
    if current_page == 'upload':
        render_upload_page()
    elif current_page == 'results':
        render_results_page()
    elif current_page == 'device_detail':
        render_device_detail_page()
    elif current_page == 'all_details':
        render_all_details_page()
    else:
        st.session_state.page = 'upload'
        st.rerun()


if __name__ == "__main__":
    main()
