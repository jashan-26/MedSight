"""
PDF Generator for Patient-Safe Medical AI Screening Reports.
ReportLab PDF engine with Navy Blue & Warm Cream Clinical Theme.
"""

import io
import os
import tempfile
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MEDICAL_DISCLAIMER, DISEASE_CLASSES


def generate_pdf_report(analysis_results, original_img_np, overlay_img_np, patient_id="PATIENT-92811", scan_name="chest_xray_scan.png"):
    """
    Generates a professional PDF medical screening report in Navy Blue & Cream theme.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Navy Blue & Cream Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#070D1E'),
        alignment=0
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#475569')
    )

    section_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0A1128'),
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B')
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#991B1B')
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("PACS AI RADIOLOGY SCREENING REPORT", title_style))
    story.append(Paragraph("Computer-Aided Triage & Explainable Decision Support Workstation • Navy & Cream Edition", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor('#070D1E'), spaceAfter=15))

    # 2. Patient & Scan Metadata Table (Navy Header & Soft Cream Rows)
    meta_data = [
        [Paragraph("<b>Patient ID:</b> " + patient_id, body_style), Paragraph("<b>Scan File:</b> " + scan_name, body_style)],
        [Paragraph("<b>Model Backbone:</b> " + analysis_results["model_name"], body_style), Paragraph("<b>Triage Risk Priority:</b> <font color='#1E3A8A'><b>" + analysis_results["risk_level"] + " RISK</b></font>", body_style)],
        [Paragraph("<b>Primary Finding:</b> <font color='#0A1128'><b>" + analysis_results["primary_finding"] + "</b></font>", body_style), Paragraph("<b>Model Confidence:</b> " + str(analysis_results["confidence_pct"]) + "%", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[3.6 * inch, 3.6 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FAF6EE')),
        ('PADDING', (0, 0), (-1, -1), 7),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#EADBC8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # 3. Save side-by-side images temporarily for ReportLab
    temp_orig = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_cam = tempfile.NamedTemporaryFile(suffix='.png', delete=False)

    try:
        Image.fromarray(original_img_np.astype(np.uint8)).save(temp_orig.name)
        Image.fromarray(overlay_img_np.astype(np.uint8)).save(temp_cam.name)

        rl_orig = RLImage(temp_orig.name, width=3.4 * inch, height=3.4 * inch)
        rl_cam = RLImage(temp_cam.name, width=3.4 * inch, height=3.4 * inch)

        img_table = Table([
            [Paragraph("<b>Preprocessed Chest Radiograph</b>", body_style), Paragraph("<b>Grad-CAM Explainable AI Overlay</b>", body_style)],
            [rl_orig, rl_cam]
        ], colWidths=[3.6 * inch, 3.6 * inch])

        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(img_table)
    finally:
        pass

    story.append(Spacer(1, 15))

    # 4. Multi-Label Probabilities Breakdown Table
    story.append(Paragraph("Multi-Label Pathology Screening Summary", section_heading))

    prob_headers = [Paragraph("<b>Pathology Condition</b>", ParagraphStyle('H1', parent=body_style, textColor=colors.HexColor('#FDFBF7'))), 
                    Paragraph("<b>Probability Score</b>", ParagraphStyle('H2', parent=body_style, textColor=colors.HexColor('#FDFBF7'))), 
                    Paragraph("<b>Risk Status</b>", ParagraphStyle('H3', parent=body_style, textColor=colors.HexColor('#FDFBF7')))]
    prob_rows = [prob_headers]

    for disease in DISEASE_CLASSES:
        prob_val = analysis_results["probabilities"].get(disease, 0.0)
        pct_str = f"{prob_val * 100:.1f}%"
        
        status_color = "#16A34A" if prob_val < 0.3 else ("#D97706" if prob_val < 0.6 else "#DC2626")
        status_text = "LOW RISK" if prob_val < 0.3 else ("MODERATE" if prob_val < 0.6 else "HIGH RISK")

        prob_rows.append([
            Paragraph(disease, body_style),
            Paragraph(pct_str, body_style),
            Paragraph(f"<font color='{status_color}'><b>{status_text}</b></font>", body_style)
        ])

    prob_table = Table(prob_rows, colWidths=[2.6 * inch, 2.3 * inch, 2.3 * inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#070D1E')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#EADBC8')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FAF6EE'), colors.HexColor('#FFFFFF')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 15))

    # 5. Anatomical Attention & Explainability
    story.append(Paragraph("Clinical Decision Support & Explainability Insights", section_heading))
    insights_text = f"""
    <b>Anatomical Focus Region:</b> {analysis_results['anatomical_attention']}<br/>
    <b>Confidence Rating:</b> {analysis_results['uncertainty']['confidence_rating']} ({analysis_results['confidence_pct']}%)<br/>
    <b>Predictive Uncertainty Entropy:</b> {analysis_results['uncertainty']['predictive_entropy']} bits<br/>
    <b>Clinical Recommendation:</b> Verification by a licensed radiologist required prior to clinical action.
    """
    story.append(Paragraph(insights_text, body_style))
    story.append(Spacer(1, 15))

    # 6. Disclaimer Box
    disclaimer_html = f"<b>CLINICAL DISCLAIMER:</b> {MEDICAL_DISCLAIMER.replace('**', '')}"
    disclaimer_table = Table([[Paragraph(disclaimer_html, disclaimer_style)]], colWidths=[7.2 * inch])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF2F2')),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#EF4444')),
    ]))
    story.append(disclaimer_table)

    # Build PDF
    doc.build(story)

    # Cleanup temp image files
    try:
        os.unlink(temp_orig.name)
        os.unlink(temp_cam.name)
    except Exception:
        pass

    buffer.seek(0)
    return buffer.getvalue()
