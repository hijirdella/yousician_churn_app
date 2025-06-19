
import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model/churn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.title("🎸 Yousician Churn Prediction App")
st.markdown("Evaluasi churn pengguna berdasarkan data aktivitas mereka dalam aplikasi pembelajaran musik.")

tab1, tab2 = st.tabs(["🧍 Single User", "📄 Batch Prediction"])

# =========================
# 1️⃣ SINGLE USER
# =========================
with tab1:
    st.subheader("Input Data Pengguna")

    n_sessions = st.number_input("Jumlah Sesi", min_value=0)
    n_exercises = st.number_input("Jumlah Latihan Unik", min_value=0)
    avg_difficulty = st.slider("Rata-rata Tingkat Kesulitan", 0.0, 10.0, 5.0)
    avg_time_playing = st.number_input("Rata-rata Durasi Bermain (detik)", min_value=0.0)
    avg_notes_eval = st.number_input("Rata-rata Not Dievaluasi", min_value=0.0)
    avg_notes_succ = st.number_input("Rata-rata Not Sukses", min_value=0.0)
    avg_chords_eval = st.number_input("Rata-rata Chord Dievaluasi", min_value=0.0)
    avg_chords_succ = st.number_input("Rata-rata Chord Sukses", min_value=0.0)
    success_ratio = st.slider("Rasio Keberhasilan Sesi", 0.0, 1.0, 0.5)
    play_mode_ratio = st.slider("Proporsi Mode 'Play'", 0.0, 1.0, 0.5)
    full_play_ratio = st.slider("Proporsi Lagu Dimainkan Penuh", 0.0, 1.0, 0.5)

    input_data = pd.DataFrame([{
        'n_sessions': n_sessions,
        'n_exercises': n_exercises,
        'avg_difficulty': avg_difficulty,
        'avg_time_playing': avg_time_playing,
        'avg_notes_eval': avg_notes_eval,
        'avg_notes_succ': avg_notes_succ,
        'avg_chords_eval': avg_chords_eval,
        'avg_chords_succ': avg_chords_succ,
        'success_ratio': success_ratio,
        'play_mode_ratio': play_mode_ratio,
        'full_play_ratio': full_play_ratio
    }])

    if st.button("🔍 Prediksi"):
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        st.subheader("Hasil Prediksi:")
        if prediction == 1:
            st.error(f"⚠️ Diprediksi **CHURN** – Probabilitas: {prob:.2%}")
        else:
            st.success(f"✅ Diprediksi **TIDAK CHURN** – Probabilitas: {1 - prob:.2%}")

# =========================
# 2️⃣ BATCH PREDICTION
# =========================
with tab2:
    st.subheader("Upload File CSV")

    st.markdown("Pastikan CSV memiliki kolom yang sama dengan input fitur.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            scaled_batch = scaler.transform(df)
            preds = model.predict(scaled_batch)
            probs = model.predict_proba(scaled_batch)[:, 1]

            result = df.copy()
            result['churn_prediction'] = preds
            result['churn_probability'] = probs

            st.dataframe(result)
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil Prediksi", data=csv, file_name='prediksi_churn.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
