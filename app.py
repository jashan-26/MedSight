"""
🩺 Medical Image Screening Assistant & Radiology Decision-Support Platform.
Professional PACS AI Workstation with Navy Blue & Cream Clinical Interface.
"""

import os
import io
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import internal modular components
from config import (
    DISEASE_CLASSES, AVAILABLE_MODELS, RISK_LEVELS,
    MEDICAL_DISCLAIMER, ANATOMICAL_MAP
)
from processing.preprocessor import ImagePreprocessor
from models.model_manager import ModelManager
from reporting.pdf_generator import generate_pdf_report
from utils.sample_generator import get_sample_xrays
from utils.metrics_store import (
    MODEL_PERFORMANCE_METRICS, DATASET_DISTRIBUTION,
    CONFUSION_MATRIX_EFFICIENTNET, get_roc_curve_data,
    get_precision_recall_curve_data
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical AI PACS Screening Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Navy Blue & Cream Custom CSS Styling
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model_manager():
    return ModelManager()


@st.cache_data
def load_sample_xrays():
    return get_sample_xrays()


def main():
    model_mgr = load_model_manager()
    sample_xrays = load_sample_xrays()
    preprocessor = ImagePreprocessor()

    # -------------------------------------------------------------------------
    # TOP PACS HEADER BANNER (NAVY BLUE & WARM CREAM ACCENTS)
    # -------------------------------------------------------------------------
    st.markdown("""
        <div class="med-header">
            <div class="header-flex">
                <div class="header-title-box">
                    <h1>🩺 PACS AI <span class="highlight">Radiology Triage</span></h1>
                    <p>Computer-Aided Multi-Label Screening & Explainable Decision-Support System</p>
                </div>
                <div class="header-badge-group">
                    <div class="pacs-status-pill">
                        <span class="status-dot-pulse"></span>
                        PACS NODE-01 • AI ONLINE
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Clinical Disclaimer Notice
    st.markdown("""
        <div class="disclaimer-banner">
            <strong>⚠️ CLINICAL TRIAGE NOTICE:</strong> This AI decision-support workstation is intended for research, screening, and emergency triage prioritization only.
            It does NOT replace diagnostic evaluation by a licensed radiologist or medical professional.
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS (NAVY BLUE THEMED PANEL)
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### ⚙️ Screening Configuration")
    
    # Model Architecture Selector
    selected_model_name = st.sidebar.selectbox(
        "Select Backbone Network",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Choose deep learning architecture for multi-label thoracic disease screening."
    )
    model_info = AVAILABLE_MODELS[selected_model_name]
    st.sidebar.info(f"**Architecture:** {selected_model_name}\n\n**Parameters:** {model_info['params']}\n\n{model_info['description']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Preprocessing Controls")
    auto_crop = st.sidebar.checkbox("Crop Dark Margins (ROI)", value=True, help="Auto-detect thoracic ROI border and crop black margins.")
    enable_clahe = st.sidebar.checkbox("CLAHE Contrast Boost", value=False, help="Enhance parenchymal lung opacities with Contrast Limited Adaptive Histogram Equalization.")
    brightness_val = st.sidebar.slider("Brightness Scale", 0.5, 1.5, 1.0, 0.05)
    contrast_val = st.sidebar.slider("Contrast Scale", 0.5, 1.5, 1.0, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔴 Grad-CAM Heatmap Options")
    colormap_choice = st.sidebar.selectbox("Colormap", ["JET", "VIRIDIS", "TURBO", "INFERNO"], index=0)
    alpha_blend = st.sidebar.slider("Overlay Blend Opacity", 0.1, 0.9, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>Navy Blue & Cream Edition v2.5<br/>PACS AI Workstation</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MAIN WORKSTATION TABS
    # -------------------------------------------------------------------------
    tab_screening, tab_analytics, tab_batch, tab_docs = st.tabs([
        "🏥 AI Radiographic Screening",
        "📊 Model Analytics & Benchmark",
        "📁 Batch Patient Triage",
        "🧠 System Architecture & Docs"
    ])

    # =========================================================================
    # TAB 1: INTERACTIVE RADIOGRAPHIC SCREENING
    # =========================================================================
    with tab_screening:
        st.markdown("#### 1. Input Patient Radiograph Scan")
        
        col_input1, col_input2 = st.columns([1, 1])

        with col_input1:
            uploaded_file = st.file_uploader(
                "Upload Chest X-Ray Scan (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"]
            )

        with col_input2:
            sample_choice = st.selectbox(
                "Or Select a Pre-Loaded Sample Radiograph:",
                ["None"] + list(sample_xrays.keys())
            )

        target_img = None
        input_source_name = "Uploaded_Scan.png"

        if uploaded_file is not None:
            target_img = Image.open(uploaded_file).convert("RGB")
            input_source_name = uploaded_file.name
        elif sample_choice != "None":
            sample_path = sample_xrays[sample_choice]
            target_img = Image.open(sample_path).convert("RGB")
            input_source_name = os.path.basename(sample_path)

        if target_img is not None:
            # Preprocess Image
            proc_pil, proc_np, input_tensor = preprocessor.preprocess(
                target_img,
                auto_crop=auto_crop,
                enable_clahe=enable_clahe,
                brightness=brightness_val,
                contrast=contrast_val
            )

            # Deep Learning Model Inference
            with st.spinner("Executing PyTorch neural inference and calculating Grad-CAM attention activation maps..."):
                results = model_mgr.predict(
                    input_tensor,
                    model_name=selected_model_name,
                    original_img_np=proc_np,
                    colormap_choice=colormap_choice,
                    alpha_blend=alpha_blend
                )

            st.markdown("---")
            st.markdown("#### 2. Clinical Screening & Triage Overview")

            # Main Diagnostic KPI Display Cards (Navy & Cream Palette)
            res_col1, res_col2, res_col3, res_col4 = st.columns([1.2, 1, 1, 1])

            primary_finding = results["primary_finding"]
            conf_pct = results["confidence_pct"]
            risk_lvl = results["risk_level"]
            unc_rating = results["uncertainty"]["confidence_rating"]

            with res_col1:
                st.markdown(f"""
                    <div class="med-card">
                        <div class="med-card-title">PRIMARY FINDING</div>
                        <div class="med-card-value" style="color: #EADBC8;">{primary_finding}</div>
                        <div class="med-card-sub">Top Predicted Multi-Label Class</div>
                    </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                    <div class="med-card">
                        <div class="med-card-title">CONFIDENCE</div>
                        <div class="med-card-value" style="color: #38BDF8;">{conf_pct}%</div>
                        <div class="med-card-sub">Rating: {unc_rating}</div>
                    </div>
                """, unsafe_allow_html=True)

            with res_col3:
                st.markdown(f"""
                    <div class="med-card">
                        <div class="med-card-title">TRIAGE PRIORITY</div>
                        <div style="margin-top: 6px;"><span class="risk-badge risk-{risk_lvl}">{risk_lvl} RISK</span></div>
                        <div class="med-card-sub">Emergency Priority Queue</div>
                    </div>
                """, unsafe_allow_html=True)

            with res_col4:
                st.markdown(f"""
                    <div class="med-card">
                        <div class="med-card-title">BACKBONE MODEL</div>
                        <div class="med-card-value" style="font-size: 1.25rem; color: #FDFBF7;">{selected_model_name}</div>
                        <div class="med-card-sub">{model_info['params']} Parameters</div>
                    </div>
                """, unsafe_allow_html=True)

            # Flag Warning Box for Low Confidence or Ambiguous Scans
            if results["uncertainty"]["low_confidence_flag"]:
                st.markdown(f"""
                    <div class="low-confidence-box">
                        ⚠️ <strong>AMBIGUOUS SCAN ALERT:</strong> {results['uncertainty']['warning_message']}
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            # Pathology Probabilities & Grad-CAM Visualizer Layout
            view_col1, view_col2 = st.columns([1, 1.2])

            with view_col1:
                st.markdown("##### 📊 Multi-Label Pathology Breakdown")
                
                prob_data = []
                for disease in DISEASE_CLASSES:
                    score = results["probabilities"][disease]
                    prob_data.append({"Pathology": disease, "Probability": score, "Percentage": f"{score*100:.1f}%"})

                df_probs = pd.DataFrame(prob_data)

                # Plotly Horizontal Bar Chart in Navy & Cream Color Palette
                fig_bar = px.bar(
                    df_probs,
                    x="Probability",
                    y="Pathology",
                    orientation="h",
                    text="Percentage",
                    color="Probability",
                    color_continuous_scale=[[0, "#131E3A"], [0.5, "#2563EB"], [1.0, "#EADBC8"]],
                    range_x=[0, 1.0]
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#FDFBF7", size=12),
                    height=290,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(234, 219, 200, 0.15)", zerolinecolor="rgba(234, 219, 200, 0.2)"),
                    yaxis=dict(gridcolor="rgba(234, 219, 200, 0.15)")
                )
                fig_bar.update_traces(textposition="outside", textfont=dict(color="#EADBC8", size=11))
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown(f"**🔍 Anatomical Attention:** <span style='color: #EADBC8;'>{results['anatomical_attention']}</span>", unsafe_allow_html=True)

            with view_col2:
                st.markdown("##### 🔴 Explainable AI Visualizer (Grad-CAM)")
                
                img_sub1, img_sub2, img_sub3 = st.columns([1, 1, 1])

                with img_sub1:
                    st.image(proc_np, caption="Preprocessed Radiograph", use_container_width=True)

                with img_sub2:
                    st.image(results["colored_heatmap"], caption="Grad-CAM Activation", use_container_width=True)

                with img_sub3:
                    st.image(results["overlay_image"], caption="Fused Attention Overlay", use_container_width=True)

            st.markdown("---")
            st.markdown("#### 3. Clinical Summary & Downloadable Report")
            
            report_col1, report_col2 = st.columns([2, 1])

            with report_col1:
                st.markdown(f"""
                <div style="background: #131E3A; border: 1px solid #2E436E; padding: 1.2rem; border-radius: 12px; color: #FAF6EE;">
                    <h5 style="color: #EADBC8; margin-top: 0;">📋 Screening Case Summary</h5>
                    <ul style="margin-bottom: 0; padding-left: 20px; line-height: 1.8;">
                        <li><strong>Patient File Source:</strong> <code>{input_source_name}</code></li>
                        <li><strong>Primary Diagnosed Class:</strong> <span style="color: #38BDF8; font-weight: 700;">{primary_finding}</span> ({conf_pct}% Confidence)</li>
                        <li><strong>Calibrated Triage Category:</strong> <span class="risk-badge risk-{risk_lvl}">{risk_lvl} RISK</span></li>
                        <li><strong>Deep Learning Backbone:</strong> <code>{selected_model_name}</code> ({model_info['params']} Parameters)</li>
                        <li><strong>Anatomical Focus:</strong> {results['anatomical_attention']}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with report_col2:
                # Generate ReportLab PDF Buffer
                pdf_bytes = generate_pdf_report(
                    analysis_results=results,
                    original_img_np=proc_np,
                    overlay_img_np=results["overlay_image"],
                    patient_id="PATIENT-92811",
                    scan_name=input_source_name
                )

                st.markdown("<br/>", unsafe_allow_html=True)
                st.download_button(
                    label="📄 Download PDF Clinical Report",
                    data=pdf_bytes,
                    file_name=f"PACS_AI_Report_{primary_finding}.pdf",
                    mime="application/pdf",
                    help="Export printable clinical PDF report in matching Navy Blue & Cream format."
                )

        else:
            st.info("👈 Please upload a Chest X-ray scan or choose a pre-loaded sample radiograph from the controls above to initiate AI screening.")

    # =========================================================================
    # TAB 2: MODEL ANALYTICS & BENCHMARKING
    # =========================================================================
    with tab_analytics:
        st.markdown("#### 📊 Comparative Deep Learning Model Evaluation")
        st.caption("Validated on the NIH ChestX-ray14 benchmark dataset (N=25,596 test partition).")

        # Performance Table
        df_metrics = pd.DataFrame(MODEL_PERFORMANCE_METRICS).T
        st.table(df_metrics[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Inference Time (ms)", "Parameters"]])

        an_col1, an_col2 = st.columns([1, 1])

        with an_col1:
            st.markdown("##### Multi-Class Receiver Operating Characteristic (ROC)")
            roc_data = get_roc_curve_data(selected_model_name)
            
            fig_roc = go.Figure()
            fig_roc.add_shape(type='line', line=dict(dash='dash', color='#64748B'), x0=0, x1=1, y0=0, y1=1)

            colors_roc = ["#EADBC8", "#38BDF8", "#F59E0B", "#22C55E", "#EC4899"]

            for idx, (disease, curve_info) in enumerate(roc_data.items()):
                fig_roc.add_trace(go.Scatter(
                    x=curve_info["fpr"],
                    y=curve_info["tpr"],
                    name=f"{disease} (AUC = {curve_info['auc']:.3f})",
                    mode='lines',
                    line=dict(width=2.5, color=colors_roc[idx % len(colors_roc)])
                ))

            fig_roc.update_layout(
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#FDFBF7"),
                height=380,
                legend=dict(x=0.45, y=0.15, bgcolor="rgba(19, 30, 58, 0.8)", bordercolor="#2E436E", borderwidth=1),
                xaxis=dict(gridcolor="rgba(234, 219, 200, 0.15)"),
                yaxis=dict(gridcolor="rgba(234, 219, 200, 0.15)")
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with an_col2:
            st.markdown(f"##### Multi-Class Confusion Matrix ({selected_model_name})")
            
            # Custom Navy to Cream Heatmap Color Ramps
            navy_cream_colorscale = [
                [0.0, "#0A1128"],
                [0.25, "#131E3A"],
                [0.5, "#1E3A8A"],
                [0.75, "#2563EB"],
                [1.0, "#EADBC8"]
            ]

            fig_cm = px.imshow(
                CONFUSION_MATRIX_EFFICIENTNET,
                x=DISEASE_CLASSES,
                y=DISEASE_CLASSES,
                color_continuous_scale=navy_cream_colorscale,
                text_auto=True
            )
            fig_cm.update_layout(
                xaxis_title="Predicted Pathology Class",
                yaxis_title="Ground Truth Pathology Class",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#FDFBF7"),
                height=380
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        st.markdown("##### ⚖️ Training Dataset Distribution & Positive Loss Weighting")
        df_dist = pd.DataFrame(list(DATASET_DISTRIBUTION.items()), columns=["Condition", "Sample Count"])
        fig_dist = px.bar(df_dist, x="Condition", y="Sample Count", color="Condition", color_discrete_sequence=["#EADBC8", "#38BDF8", "#2563EB", "#F59E0B", "#22C55E"])
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#FDFBF7"),
            height=320,
            xaxis=dict(gridcolor="rgba(234, 219, 200, 0.15)"),
            yaxis=dict(gridcolor="rgba(234, 219, 200, 0.15)")
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # =========================================================================
    # TAB 3: BATCH PATIENT TRIAGE
    # =========================================================================
    with tab_batch:
        st.markdown("#### 📁 Emergency Batch Radiograph Triage Workstation")
        st.caption("Upload multiple patient scans to automatically organize an emergency priority triage queue.")

        batch_files = st.file_uploader(
            "Upload Batch Chest X-Ray Files",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if batch_files:
            batch_results = []
            
            with st.spinner(f"Screening batch queue ({len(batch_files)} patient scans)..."):
                for idx, b_file in enumerate(batch_files):
                    img = Image.open(b_file).convert("RGB")
                    _, b_np, b_tensor = preprocessor.preprocess(img)
                    res = model_mgr.predict(b_tensor, model_name=selected_model_name, original_img_np=b_np)
                    
                    batch_results.append({
                        "Patient File": b_file.name,
                        "Primary Finding": res["primary_finding"],
                        "Confidence Score": f"{res['confidence_pct']}%",
                        "Risk Category": res["risk_level"],
                        "Ambiguity Flag": "⚠️ Low Confidence" if res["uncertainty"]["low_confidence_flag"] else "Validated",
                        "Triage Status": "STAT URGENT" if res["risk_level"] == "HIGH" else ("EVALUATE" if res["risk_level"] == "MODERATE" else "ROUTINE")
                    })

            df_batch = pd.DataFrame(batch_results)
            
            # Sort by High Risk priority
            df_batch["Priority_Index"] = df_batch["Risk Category"].apply(lambda x: 1 if x == "HIGH" else (2 if x == "MODERATE" else 3))
            df_batch = df_batch.sort_values("Priority_Index").drop(columns=["Priority_Index"])

            st.markdown("##### 🚨 Triage Priority Queue Output")
            st.dataframe(df_batch, use_container_width=True)

            csv_buffer = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Triage Summary CSV",
                data=csv_buffer,
                file_name="pacs_batch_triage_results.csv",
                mime="text/csv"
            )
        else:
            st.info("Upload multiple X-ray scans above to generate automated emergency triage priority queues.")

    # =========================================================================
    # TAB 4: SYSTEM ARCHITECTURE & DOCS
    # =========================================================================
    with tab_docs:
        st.markdown("#### 🧠 End-to-End System Pipeline & Technology Stack")
        
        st.markdown("""
        <div style="background: #131E3A; border: 1px solid #2E436E; padding: 1.5rem; border-radius: 12px; font-family: 'JetBrains Mono', monospace; color: #EADBC8;">
        CHEST X-RAY INPUT RAD SCAN<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;▼<br/>
        Image Preprocessing Subsystem (Otsu ROI Crop + CLAHE Contrast Enhancement)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;▼<br/>
        PyTorch Deep Learning Backbone (EfficientNet-B0 / ResNet-50 Feature Extractor)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;├──────────────────────────┐<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;▼                          ▼<br/>
        Multi-Label Probabilities      Grad-CAM Activation Gradient Hooking<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;│                          │<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;▼                          ▼<br/>
        Temperature Scaling            Anatomical Attention Heatmap Overlay<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;│                          │<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;└────────────┬─────────────┘<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PACS AI Triage Screening & PDF Clinical Report<br/>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(MEDICAL_DISCLAIMER)


if __name__ == "__main__":
    main()
