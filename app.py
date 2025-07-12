import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Load your trained logistic regression model
model = joblib.load("personality_model.pkl")

# Set Streamlit page config
st.set_page_config(page_title="🧠 Personality Predictor", page_icon="🌀", layout="centered")

# 🌟 HEADER with emojis and styled text
st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <h1 style='color:#6a1b9a;'>🧠 Personality Prediction App</h1>
        <h4 style='color:#616161;'>Powered by <span style='color:#2e7d32;'>Logistic Regression</span></h4>
        <p style='font-size:18px;'>Predict if someone is an <strong>Introvert</strong> or <strong>Extrovert</strong> based on behavioral traits.</p>
    </div>
    <hr style="border: 1px solid #ddd;">
""", unsafe_allow_html=True)

# 🎚️ User Input Section
st.subheader("📝 Enter Personality Traits Below:")

col1, col2 = st.columns(2)

with col1:
    social_time = st.slider("🕒 Social time per day (hours)", 0, 8, 4)
    talkative_score = st.slider("💬 Talkative score (1-10)", 1, 10, 5)
    crowd_comfort = st.slider("👥 Comfort in crowds (1-10)", 1, 10, 5)
    num_friends = st.slider("👫 Number of close friends", 0, 15, 5)

with col2:
    prefers_solo = st.selectbox("🧘 Prefers solo activities?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    energy_after_social = st.selectbox("⚡ Energy after socializing", ["Energized (0)", "Drained (1)"])
    public_speaking = st.selectbox("🎤 Enjoy public speaking?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    decision_speed = st.selectbox("🚀 Decision making speed", [0, 1], format_func=lambda x: "Fast" if x == 0 else "Slow")

# Convert selectbox to numeric
energy_after_social = 0 if energy_after_social == "Energized (0)" else 1

# 🎯 Predict Button
if st.button("🔍 Predict Personality"):
    # Input features as array
    input_features = np.array([[social_time, talkative_score, prefers_solo, crowd_comfort,
                                energy_after_social, num_friends, public_speaking, decision_speed]])

    prediction = model.predict(input_features)[0]

    st.markdown("---")

    # ✅ Prediction Result
    if prediction == 1:
        st.success("🟢 The person is likely an **Extrovert** 🎉\n\nThey enjoy social interaction, are expressive, and energized by people.")
    else:
        st.warning("🔵 The person is likely an **Introvert** 🌙\n\nThey prefer quiet environments, deep thinking, and solo activities.")

    # 📊 Model Info
    st.info("📌 Model Used: **Logistic Regression**")

    # 📈 Show input traits as Bar Chart
    st.subheader("📊 Trait Scores Overview")

    feature_names = [
        "Social Time", "Talkative", "Prefers Solo", "Crowd Comfort",
        "Energy After Social", "Close Friends", "Public Speaking", "Decision Speed"
    ]

    st.bar_chart(data=np.array(input_features[0]), use_container_width=True)

    # 🕸️ Radar Chart
    st.subheader("🕸️ Personality Radar Chart")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=input_features[0],
        theta=feature_names,
        fill='toself',
        name='Personality Traits',
        line=dict(color='royalblue')
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# 🌟 Footer
st.markdown("""
    <hr>
    <div style='text-align: center; color: gray;'>
        <small>💻 Built with ❤️ using <strong>Streamlit</strong> | © 2025 Personality AI</small>
    </div>
""", unsafe_allow_html=True)
