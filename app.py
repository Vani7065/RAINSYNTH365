import streamlit as st
import numpy as np
import tensorflow as tf
import traceback

st.title("💧 Runoff Prediction using LSTM")
st.markdown("Upload 1 year (365 days) of forcing data to predict next day streamflow.")

# Try loading the model with logging
try:
    model = tf.keras.models.load_model("lstm_runoff_model.h5")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model.")
    st.code(traceback.format_exc())
    st.stop()

uploaded_file = st.file_uploader("Upload forcing data (.npy or .csv)", type=["npy", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".npy"):
            data = np.load(uploaded_file)
        else:
            data = np.loadtxt(uploaded_file, delimiter=',')

        if data.shape[0] != 365:
            st.error(f"❌ Expected 365 rows, got {data.shape[0]}.")
        elif data.ndim != 2 or data.shape[1] != 8:
            st.error(f"❌ Expected 8 features per row, got {data.shape[1]}.")
        else:
            input_sequence = np.expand_dims(data, axis=0)
            prediction = model.predict(input_sequence)
            st.success(f"✅ Predicted Streamflow: {prediction[0][0]:.2f}")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.code(traceback.format_exc())
