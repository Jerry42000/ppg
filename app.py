import streamlit as st
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="PPG Monitor", layout="wide")
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False
if "agreed" not in st.session_state:
    st.session_state.agreed = False

# PPG explainer
if not st.session_state.intro_done:
    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            "<h1 style='font-size:3.0rem; margin-bottom:0.4em;'>📈 What is a PPG?</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style='font-size:1.15rem; line-height:1.6;'>
            <p><strong>PPG (Photoplethysmography)</strong> is a way to measure tiny changes in blood volume in your
            fingertip using your phone’s camera and light. Each heartbeat pushes blood through your finger; the
            camera picks up those rhythmic changes.</p>

            <h3>What this app estimates from your video</h3>
            <ul>
              <li><strong>Heart rate (BPM)</strong> — beats per minute.</li>
              <li><strong>HRV metrics</strong> — how regular/irregular the time between beats is:
                  SDNN, RMSSD, and pNN50.</li>
              <li><strong>LF/HF ratio</strong> — a rough frequency-domain index sometimes used in HRV research.</li>
              <li><strong>Rhythm classification (experimental)</strong> — a conservative rule-based check
                  for patterns like sinus rhythm vs. irregular rhythms.</li>
            </ul>

            <h3>How it works</h3>
            <ol>
              <li>You cover the camera/light with a fingertip and record ~90–120 seconds of video.</li>
              <li>We track the subtle red-channel intensity changes over time → that’s your PPG waveform.</li>
              <li>We find peaks (heartbeats), compute beat-to-beat intervals (IBIs), then derive HR/HRV metrics.</li>
            </ol>

            <h3>Limitations & safety</h3>
            <ul>
              <li>PPG is <em>not</em> an ECG (Electrocardiogram). It infers timing from optical changes, which is sensitive to motion,
                  lighting, and skin contact.</li>
              <li>Results are for <strong>reference only</strong>. If you feel unwell, please seek medical care.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        with right:
            spacer_px = 240  
            st.markdown(f"<div style='height:{spacer_px}px'></div>", unsafe_allow_html=True)
        r_l, r_c, r_r = st.columns([1, 3, 1])
        with r_c:
            img_path = "fig3.png"  
            bigger_width = 600             

            if os.path.exists(img_path):
                st.image(img_path, caption="Source: P.H. Charlton", width=bigger_width)
            else:
                st.info(
                    "Add **ppg_explainer.png** to this folder (or change the file name in app.py) "
                    "to display an illustration here."
                )

    # Centered continue button
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        if st.button("➡️ Continue to Instructions", use_container_width=True):
            st.session_state.intro_done = True
    st.stop()

# Instruction
if "agreed" not in st.session_state:
    st.session_state.agreed = False 

if not st.session_state.agreed:
    col_left, col_right = st.columns(2)
    # Title
    with col_left:
        st.markdown(
            "<h1 style='font-size:3.5rem; margin-bottom:0.2em;'>📝 Instructions</h1>",
            unsafe_allow_html=True
        )
        # Intro
        st.markdown(
            "<p style='font-size:1.25rem; margin-top:0;'>"
            "Please read the following before using the PPG Heart-Rate & HRV Monitor:"
            "</p>",
            unsafe_allow_html=True
        )

        # Merged
        st.markdown(
            """
            <ol style='font-size:1.1rem; line-height:1.6;'>
            <li>
                <strong><u>Positioning Your Hands</u></strong>
                <ul>
                <li>Place your fingertip <strong>directly</strong> over the center of the camera’s light or flash.</li>
                <li>Keep your finger <strong>flat</strong> against the lens—no gaps or tilted angles. See Figure 1.</li>
                <li>Make sure your fingernail is <strong>facing away</strong> from the lens so light penetrates the flesh.</li>
                <li>You should see a uniform <strong>red</strong> image from the screen. See Figure 2.</li>
                </ul>
            </li>
            <li>
                <strong><u>Recording the Video</u></strong>
                <ul>
                <li>Use your phone’s <strong>back</strong> camera (not the selfie camera) if possible.</li>
                <li>Set resolution to at least <strong>720p</strong> (higher is OK).</li>
                <li>Hold the record button steady and <strong>don’t tap the screen</strong> while filming.</li>
                <li>Save in <strong>MP4</strong> or <strong>AVI</strong> format—keep the file under 200 MB.</li>
                </ul>
            </li>
            <li>
                <strong><u>Lighting</u></strong><br>
                Record in a <strong>dark or dim room</strong> is preferred.
                Please at least ensure even, moderate brightness.
                <ul>
                <li>No direct sunlight or glare, and no talking during recording.</li>
                </ul>
            </li>
            <li>
                <strong><u>Stillness</u></strong><br>
                Rest your arm on a stable surface and keep your hand as motionless as possible.
            </li>
            <li>
                <strong><u>Duration</u></strong><br>
                Record at least <strong>90 s</strong> for reliable HRV metrics; <strong>2 min</strong> preferred.
            </li>
            <li>
                <strong><u>Safety</u></strong><br>
                This tool, especially the Rhythm Prediction, is for <strong>reference</strong> only and not a medical diagnosis. Always consult a medical professional if you feel unwell.
            </li>
            <li>
                <strong><u>Data Privacy</u></strong><br>
                Your video will be processed locally and not stored or shared with anyone. Please refrain from sharing any personal information during the recording.
            </li>
            <li>
                <strong><u>Feedback</u></strong><br>
                Thank you for using this app! If you have any issues or suggestions, please contact me at <a href="mailto:wensong.dai@mail.utoronto.ca">wensong.dai@mail.utoronto.ca</a>.
            </ol>
            """,
            unsafe_allow_html=True
        )
    with col_right:
        st.image("fig1.png", caption="**Figure 1: Finger Positioning**", use_container_width=True)                 
        blank1, center, blank2 = st.columns([1, 2, 1])
        with center:
            st.image(
                "fig2.png",
                caption="Figure 2: Correct View on Screen & Resolution",
                use_container_width=False,
                width=320,
            )
    # Continue
    btn_pad_left, btn_center, btn_pad_right = st.columns([1, 2, 1])
    with btn_center:
        if st.button("✅ Continue to PPG Monitor", use_container_width=True):
            st.session_state.agreed = True
    st.stop()  

from ppg_analysis import run_ppg_pipeline

st.set_page_config(page_title="PPG Monitor", layout="wide")
st.title("📹 Photoplethysmogram (PPG) Heart Rate & Variability Monitor")

# Sidebar
st.sidebar.header("Upload Video")
video_file = st.sidebar.file_uploader(
    "Upload a fingertip PPG video (mp4/avi)",
    type=["mp4", "avi"]
)

# Content
if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    st.sidebar.markdown(f"**File:** {os.path.basename(tmp_path)}")
    st.sidebar.markdown(f"**Size:** {os.path.getsize(tmp_path)//1024} KB")

    st.info("Running PPG analysis… this may take 10–20 s.")

    metrics, figs = run_ppg_pipeline(tmp_path)

    os.remove(tmp_path)

    c1, c2, c3, c4, c5,c_help = st.columns([1, 1, 1, 1, 1, 0.28])
    def arrow_for(value, low=None, high=None):
        if value is None or (isinstance(value, float) and (value != value)): 
            return None
        if low is not None and value < low:
            return "- "   
        if high is not None and value > high:
            return "+ "  
        return None

    c1, c2, c3, c4, c5,c_help = st.columns([1, 1, 1, 1, 1, 0.28])

    # Heart Rate
    c1.metric(
        "Heart Rate (BPM)",
        f"{metrics['bpm']:.1f}",
        delta=arrow_for(metrics['bpm'], low=50, high=100)
    )

    # SDNN
    c2.metric(
        "SDNN (ms)",
        f"{metrics['sdnn']:.1f}",
        delta=arrow_for(metrics['sdnn'], low=20, high=80)
    )

    # RMSSD
    c3.metric(
        "RMSSD (ms)",
        f"{metrics['rmssd']:.1f}",
        delta=arrow_for(metrics['rmssd'], low=20, high=100)
    )

    # pNN50
    c4.metric(
        "pNN50 (%)",
        f"{metrics['pnn50']:.1f}",
        delta=arrow_for(metrics['pnn50'], low=3, high=30)
    )

    # LF/HF ratio
    lf_hf_val = metrics.get('lf_hf', float('nan'))
    c5.metric(
        "LF/HF Ratio",
        f"{lf_hf_val:.2f}" if lf_hf_val == lf_hf_val else "—",
        delta=arrow_for(lf_hf_val, low=0.5, high=2.0) if lf_hf_val == lf_hf_val else None
    )
    with c_help:
        with st.popover("❓", use_container_width=True):
            st.markdown("### What do these mean?")
            st.markdown(
                """
                **Heart Rate (BPM)** — Beats per minute.  
                *Typical resting adults:* **50–100 BPM** (healthy trained individuals can be lower).

                **SDNN (ms)** — Standard deviation of NN intervals *(overall HRV)*.  
                *Typical resting:* **30–100 ms** (training/age dependent, need to be careful if value drops below 30ms, which may indicate unhealthy cardiac system).

                **RMSSD (ms)** — Short-term vagal HRV metric.  
                *Typical resting:* **20–80 ms** (higher with more vagal tone/fitness).

                **pNN50 (%)** — % of successive NN pairs that differ by >50 ms.  
                *Typical resting:* **3–29%** (very high may indicate irregular rhythm; very low suggests low sensitivity of autonomic nervous system, and may indicate a risk of cardiac events).

                **LF/HF Ratio** — Rough **balance** between sympathetic/parasympathetic system. A higher ratio indicates greater sympathetic activity (fight or flight), while a lower ratio indicates greater parasympathetic activity (rest and digest).
                *Normal-ish band:* **0.5–2.0** (interpretation depends on context & breathing).
                """
            )
            st.caption(
                "These are educational estimates, not medical diagnoses. "
                "If you feel unwell, seek medical care."
            )
# Rhythm classification
    rc_main, rc_spacer, rc_help2 = st.columns([5, 1, 0.28])

    with rc_main:
        st.metric("Rhythm Classification", metrics['rhythm'])

    with rc_help2:
        with st.popover("❓", use_container_width=True):
            st.markdown("""
            <style>
            /* Make the popover big, and wipe *all* top spacing */
            [data-testid="stPopoverContent"]{
                max-width: min(1100px, 95vw) !important;
                height: 80vh !important;               /* tall modal-like panel */
                overflow: auto !important;
                padding: 10px 18px 16px 18px !important;  /* small, even padding */
            }

            /* Streamlit wraps content in internal blocks — clear their top spacing */
            [data-testid="stPopoverContent"] > div:first-child{
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            [data-testid="stPopoverContent"] [data-testid="stVerticalBlock"]{
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            [data-testid="stPopoverContent"] [data-testid="stHorizontalBlock"]{
                margin-top: 0 !important;
                padding-top: 0 !important;
            }

            /* Headings inside the popover shouldn’t re-introduce a gap */
            [data-testid="stPopoverContent"] h1,
            [data-testid="stPopoverContent"] h2,
            [data-testid="stPopoverContent"] h3{
                margin-top: 0 !important;
            }
            [data-testid="stPopoverContent"] p{
                margin-top: .35rem !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("## What Types of Rhythms are There?")
            st.markdown(
                """
                **Important:** This is a PPG-based, rule-driven estimate—**not** an ECG diagnosis.
                We infer rhythm only from beat-to-beat timing variability (IBIs/HRV) and simple
                pattern checks. Motion, lighting, breathing, and arrhythmias can all affect the result.

                ### Labels you may see

                **Sinus Rhythm (normal)**
                - Beats are **regular** with small, physiologic variability (often SDNN ~20–80 ms, RMSSD ~20–80 ms, pNN50 ~3–30% at rest).
                - No repeating “short–long” patterns; no clusters of very short or very long beats.
                - Heart rate usually in the typical resting range (≈50–100 BPM), but can be higher/lower if you were moving, anxious, or athletic.

                **Sinus Tachycardia / Sinus Bradycardia**
                - Rhythm still **regular**, but the rate is **fast (>100 BPM)** or **slow (<50 BPM)**.
                - We do not try to decide *why* (exercise, fever, fitness, medication, etc.). It’s simply a rate descriptor with regular timing.

                **Atrial Fibrillation (AF) — “Irregular”**
                - AF on ECG lacks P-waves; with PPG we can only approximate by **timing irregularity**:
                - **High dispersion** of IBIs (e.g., RMSSD ↑, pNN50 ↑),  
                - **Low periodicity/autocorrelation**,  
                - **No dominant short–long alternation** (i.e., not bigeminy/trigeminy).
                - We combine those features and require “enough evidence” over a long window before showing AF.

                **PVCs / Ventricular Bigeminy / Trigeminy**
                - **PVC**: a premature beat (short RR) followed by a **compensatory pause** (long RR).  
                In our timing-only view: a cluster of **short IBI** then **long IBI** differences (e.g., <−200 ms then >+200 ms).
                - **Bigeminy**: consistent **short-long-short-long** alternation.  
                - **Trigeminy**: repeating pattern every three beats (e.g., normal-normal-PVC).
                - We detect these by checking for **stable alternation patterns** in the IBI ratio stream (e.g., “short vs. long” cadence).

                **Ventricular Fibrillation (VF) – EMERGENCY flag**
                - True VF is an ECG emergency. We **do not diagnose VF** with PPG; we only flag obviously **erratic**/non-physiologic timing
                (e.g., extremely high apparent rate, mean IBI <200 ms, or wildly unstable signal) as **“Ventricular Fibrillation — seek help”**
                to minimize the chance of missing an extreme scenario. Motion artifacts can mimic this—**trust how you feel**.
                ---

                ### How We Decide
                - Extract **peaks** → compute **inter-beat intervals (IBIs)**.  
                - Derive **HRV features** (SDNN, RMSSD, pNN50).
                - Mark *regular* vs *irregular* (AF):,
                - Detect *short–long alternation* (bigeminy/trigeminy),
                - Spot *isolated short + compensatory long waves* (PVC),
                - Check for *extremely irregular timing* (VF).

                > **Caution:** PPG ≠ ECG. Use this only for education and personal tracking.  
                > If you feel unwell, **please seek medical care.**
                """
            )

            st.markdown("</div>", unsafe_allow_html=True)
    # Display Plots
    st.subheader("Filtered PPG with Detected Beats")
    st.pyplot(figs["filtered_peaks"])

    st.subheader("Inter‑Beat Interval Distribution")
    st.pyplot(figs["ibi_hist"])

    st.success("Analysis complete!")

else:
    st.write("👈 Upload a video file on the left to begin.")
    # References
sp_l, sp_c, sp_r = st.columns([1, 2, 1])
with sp_c:
    st.markdown("<hr style='opacity:.25;'>", unsafe_allow_html=True)
    with st.popover("References & Citations", use_container_width=False):
        st.markdown("""
        <style>
        [data-testid="stPopoverContent"]{
            max-width: min(950px, 95vw) !important;
            height: 70vh !important;
            overflow: auto !important;
            padding: 12px 18px 16px 18px !important;
        }
        [data-testid="stPopoverContent"] > div:first-child{
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        [data-testid="stPopoverContent"] h2,
        [data-testid="stPopoverContent"] h3{
            margin-top: .4rem !important;
        }
        [data-testid="stPopoverContent"] p{
            margin-top: .35rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        ## References & Citations

        1. Welltory, “RMSSD, pNN50, SDNN and other HRV measurements,” Welltory, Dec. 25, 2022.
            https://welltory.com/rmssd-and-other-hrv-measurements/   


        2. T. Pereira et al., “Photoplethysmography based atrial fibrillation detection: a review,” npj Digital Medicine, vol. 3, no. 1, Jan. 2020. 
            doi: https://doi.org/10.1038/s41746-019-0207-9

        3. F. Shaffer and J. P. Ginsberg, “An Overview of Heart Rate Variability Metrics and Norms,” Frontiers in Public Health, vol. 5, no. 258, Sep. 2017.
            doi: https://doi.org/10.3389/fpubh.2017.00258

        4. X. Zhou, H. Ding, B. Ung, E. Pickwell-MacPherson, and Y. Zhang, “Automatic online detection of atrial fibrillation based on symbolic dynamics and Shannon entropy,” BioMedical Engineering OnLine, vol. 13, no. 1, p. 18, 2014. 
            doi: https://doi.org/10.1186/1475-925x-13-18.

        5. Z. Chen et al., “Prognostic implications of premature ventricular contractions and non-sustained ventricular tachycardia in light-chain cardiac amyloidosis,” Europace, vol. 26, no. 3, Mar. 2024.
            doi: https://doi.org/10.1093/europace/euae063.

        6. “Heart rate variability (HRV) - What is normal HRV range,” Kubios, Sep. 23, 2024. 
            https://www.kubios.com/blog/heart-rate-variability-normal-range/

        7. “Heart Rate Variability -,” caringmedical.com, Nov. 22, 2023.
            https://caringmedical.com/hauser-neck-center/heart-rate-variability/
        """)

    
