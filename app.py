
import streamlit as st
import plotly.express as px
import pandas as pd

from utils.pdf_reader import extract_text_from_pdf
from utils.summarizer import generate_summary
from utils.ml_risk_models import predict_risk_all, get_accuracy
from utils.chatbot import chatbot_response
from utils.report_generator import create_pdf
from utils.incident_memory import (
    find_similar_incident,
    save_incident
)
from utils.ai_recommendation import generate_recommendation
from utils.image_analyzer import analyze_image
from utils.video_analyzer import summarize_video

# PAGE SETTINGS
st.set_page_config(
    page_title="SafeBrief AI",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>

body {
    background-color: #050816;
}

.main {
    background: linear-gradient(135deg, #050816 0%, #0b1120 100%);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.metric-card {
    background: linear-gradient(135deg, #111827, #1e293b);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 15px;
}

.metric-title {
    color: #94a3b8;
    font-size: 18px;
}

.metric-value {
    font-size: 42px;
    font-weight: bold;
    color: #4ade80;
}

.big-title {
    font-size: 72px;
    font-weight: 800;
    background: linear-gradient(to right, #4ade80, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 22px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 34px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 20px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# SESSION STATE
if "generated" not in st.session_state:
    st.session_state.generated = False

# TITLE
st.markdown("""
<div>

<div class='big-title'>
SafeBrief AI
</div>

<div class='subtitle'>
AI-Powered Industrial Safety Intelligence Platform
</div>

</div>
""", unsafe_allow_html=True)

# TOP METRICS


# SIDEBAR
st.sidebar.title("Control Panel")

st.sidebar.markdown("---")

st.sidebar.success(
    "Industrial AI Safety Monitoring System"
)

st.sidebar.markdown(
    "Upload reports, images, and videos for AI-powered risk analysis."
)

# PDF UPLOAD
uploaded_file = st.sidebar.file_uploader(
    "Upload Incident Report",
    type=["pdf"]
)

# IMAGE UPLOAD
image_file = st.sidebar.file_uploader(
    "Upload Incident Image",
    type=["png", "jpg", "jpeg"]
)

# VIDEO UPLOAD
video_file = st.sidebar.file_uploader(
    "Upload Safety Video",
    type=["mp4", "mov", "avi"]
)

# GENERATE BUTTON
if st.sidebar.button("Generate Insights"):
    st.session_state.generated = True

# IMAGE ANALYSIS
if image_file:

    st.image(image_file, caption="Uploaded Incident Image")

    image_result = analyze_image(image_file)

    st.markdown(
        "<div class='section-title'>AI Image Safety Analysis</div>",
        unsafe_allow_html=True
    )

    st.markdown(image_result)

# VIDEO ANALYSIS
if video_file:

    st.video(video_file)

    video_summary = summarize_video(video_file)

    st.markdown(
        "<div class='section-title'>AI Video Incident Summary</div>",
        unsafe_allow_html=True
    )

    st.markdown(video_summary)

# MAIN PROCESS
if uploaded_file and st.session_state.generated:

    # EXTRACT TEXT
    text = extract_text_from_pdf(uploaded_file)

    # SUMMARY
    summary = generate_summary(text)

    # SAVE INCIDENT
    save_incident(
        summary,
        "Improve inspections and strengthen safety compliance."
    )

    # RISK PREDICTION
    risk_results = predict_risk_all(text)

    risk = risk_results["Final Risk"]

    # SIMILAR INCIDENT
    similar_incident = find_similar_incident(text)

    # AI RECOMMENDATION
    if not similar_incident:
        ai_solution = generate_recommendation(text)

    # COLORS
    risk_color = {
        "HIGH": "red",
        "MEDIUM": "orange",
        "LOW": "green"
    }

    # TOP SECTION
    col1, col2 = st.columns(2)

    # LEFT SIDE
    with col1:

        st.markdown(
            "<div class='section-title'>AI Incident Analysis</div>",
            unsafe_allow_html=True
        )

        sections = summary.split("ROOT CAUSE:")

        main_summary = sections[0]

        rest = sections[1] if len(sections) > 1 else ""

        hazards_split = rest.split("HAZARDS:")

        root_cause = hazards_split[0]

        hazards_rest = hazards_split[1] if len(hazards_split) > 1 else ""

        recommendation_split = hazards_rest.split("RECOMMENDATIONS:")

        hazards = recommendation_split[0]

        recommendations = recommendation_split[1] if len(recommendation_split) > 1 else ""

        st.markdown("### Summary")
        st.info(main_summary)

        st.markdown("### Root Cause")
        st.warning(root_cause)

        st.markdown("### Hazards")
        st.error(hazards)

        st.markdown("### Recommended Actions")
        st.success(recommendations)

    # RIGHT SIDE
    with col2:

        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #111827, #1e293b);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            margin-top: 10px;
        ">
        """, unsafe_allow_html=True)

        st.markdown("""
        <h2 style='
            color:white;
            font-size:40px;
            font-weight:700;
            margin-bottom:20px;
        '>
        AI Risk Intelligence
        </h2>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <h1 style='
            color:{risk_color[risk]};
            font-size:80px;
            font-weight:900;
            margin-bottom:15px;
            '>
            {risk}
            </h1>
            """,
            unsafe_allow_html=True
        )

        st.progress(85)

        st.markdown("""
        <p style='
            color:#94a3b8;
            font-size:18px;
            margin-top:10px;
        '>
        AI Confidence Score: 85%
        </p>
        """, unsafe_allow_html=True)

        st.markdown("## ML Risk Prediction Models")

        accuracy = get_accuracy()

        st.success(f"""
        ✅ Logistic Regression: {risk_results['Logistic Regression']} (Accuracy: {accuracy.get('logistic_regression', 'N/A')}%)

        ✅ Random Forest: {risk_results['Random Forest']} (Accuracy: {accuracy.get('random_forest', 'N/A')}%)

        ✅ Decision Tree: {risk_results['Decision Tree']} (Accuracy: {accuracy.get('decision_tree', 'N/A')}%)
        """)

        st.markdown("## Final Ensemble Decision")

        st.error(f"""
        ⚠️ Final Predicted Risk:

        {risk_results['Final Risk']}
        """)

        

    # EXECUTIVE INSIGHTS
    st.markdown(
        "<div class='section-title'>Executive Insights</div>",
        unsafe_allow_html=True
    )

    if risk == "HIGH":

        st.error("""
        • Immediate management attention required

        • Potential operational shutdown risk

        • Safety compliance violation detected

        • Urgent corrective action recommended
        """)

    elif risk == "MEDIUM":

        st.warning("""
        • Moderate operational risk detected

        • Preventive maintenance recommended

        • Safety review should be conducted
        """)

    else:

        st.success("""
        • Low operational risk

        • Maintain current safety standards

        • Continue periodic inspections
        """)

  

       
    # CHART
    human_error = text.lower().count("worker")
    machine_failure = text.lower().count("machine")
    maintenance = text.lower().count("maintenance")
    compliance = text.lower().count("safety")

    data = pd.DataFrame({
        "Cause": [
            "Human Error",
            "Machine Failure",
            "Maintenance",
            "Compliance"
        ],
        "Count": [
            human_error + 1,
            machine_failure + 1,
            maintenance + 1,
            compliance + 1
        ]
    })

    fig = px.pie(
        data,
        names="Cause",
        values="Count",
        title="Incident Cause Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # CHATBOT
    st.markdown(
        "<div class='section-title'>AI Safety Assistant</div>",
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Ask a question about the incident report"
    )

    if st.button("Ask AI"):

        if question.strip() != "":

            answer = chatbot_response(question, text)

            st.success(answer)

        else:

            st.warning("Please enter a question.")

    # PDF REPORT
    pdf_path = create_pdf(summary, risk)

    with open(pdf_path, "rb") as pdf_file:

        st.download_button(
            label="Download AI Report",
            data=pdf_file,
            file_name="SafeBrief_Report.pdf",
            mime="application/pdf"
        )

# FOOTER


