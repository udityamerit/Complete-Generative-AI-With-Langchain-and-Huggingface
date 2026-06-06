import streamlit as st
import numpy as np
import pickle
import time
import pandas as pd
from typing import Tuple, Optional, Any
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Model Sequence Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Enterprise CSS Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f9fafb; }
    .main-header {
        font-size: 2.2rem;
        color: #111827;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 2rem;
        margin-top: 0px;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching & Resource Loading ---
@st.cache_resource(show_spinner="Initializing Neural Networks...")
def load_keras_models() -> Tuple[Any, Any]:
    """Loads both LSTM and GRU compiled models."""
    lstm_model = load_model('LSTM_model.h5')
    gru_model = load_model('GRU_model.h5')
    return lstm_model, gru_model

@st.cache_resource(show_spinner="Initializing Tokenizer...")
def load_keras_tokenizer() -> Any:
    """Loads the trained tokenizer."""
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# --- Core Prediction Logic ---
def predict_next_word(model: Any, tokenizer: Any, text: str, max_sequence_len: int) -> Tuple[Optional[str], float]:
    """
    Processes input text and predicts the next word using the specified model.
    """
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if not token_list:
            return None, 0.0

        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
            
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_word_index = np.argmax(predicted_probs)
        confidence = float(predicted_probs[predicted_word_index] * 100)
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word, confidence
                
        return None, 0.0
    except Exception as e:
        return None, 0.0

# --- App Header ---
st.markdown('<div class="main-header">Sequence Prediction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Model Architecture Analysis (LSTM vs. GRU)</div>', unsafe_allow_html=True)

# --- Resource Initialization ---
try:
    lstm_net, gru_net = load_keras_models()
    tokenizer = load_keras_tokenizer()
    # Assuming both models were trained on the same sequence length
    max_sequence_len = lstm_net.input_shape[1] + 1
except Exception as e:
    st.error(f"System Error: Unable to locate required binaries. Ensure 'LSTM_model.h5', 'GRU.h5', and 'tokenizer.pickle' are present. Details: {e}")
    st.stop()

# --- User Interface ---
test_cases = [
    "To be or not to",
    "A picture is worth a thousand",
    "The quick brown fox jumps over the lazy"
]

with st.sidebar:
    st.markdown("### Configuration")
    selected_test = st.selectbox("Load Test Sequence:", ["Select a sequence..."] + test_cases)
    st.markdown("---")
    st.markdown("### Model Metadata")
    st.text(f"Max Sequence Length: {max_sequence_len}")
    st.text(f"Vocabulary Size: {len(tokenizer.word_index)}")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if selected_test != "Select a sequence...":
    st.session_state.input_text = selected_test

input_text = st.text_area(
    label="Input Sequence", 
    value=st.session_state.input_text, 
    height=120,
    placeholder="Enter text sequence here to evaluate both models..."
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("Run Models", type="primary", use_container_width=True)

# --- Output Generation & Tabular Comparison ---
if predict_btn:
    if not input_text.strip():
        st.warning("Input sequence cannot be empty.")
    else:
        with st.spinner("Processing through neural networks..."):
            time.sleep(0.2) 
            
            # Run predictions for both models
            lstm_word, lstm_conf = predict_next_word(lstm_net, tokenizer, input_text, max_sequence_len)
            gru_word, gru_conf = predict_next_word(gru_net, tokenizer, input_text, max_sequence_len)
            
            st.markdown("### Model Performance Analysis")
            
            # Construct the DataFrame for tabular display
            results_data = {
                "Model Architecture": ["Long Short-Term Memory (LSTM)", "Gated Recurrent Unit (GRU)"],
                "Predicted Word": [
                    lstm_word if lstm_word else "N/A (Error)", 
                    gru_word if gru_word else "N/A (Error)"
                ],
                "Confidence Score": [
                    f"{lstm_conf:.2f}%" if lstm_word else "0.00%", 
                    f"{gru_conf:.2f}%" if gru_word else "0.00%"
                ]
            }
            
            df_results = pd.DataFrame(results_data)
            
            # Display using Streamlit's native dataframe (hiding index for a cleaner look)
            st.dataframe(
                df_results, 
                use_container_width=True, 
                hide_index=True
            )
            
            # Contextual display card showing the highest confidence result
            best_word = lstm_word if lstm_conf >= gru_conf else gru_word
            winning_model = "LSTM" if lstm_conf >= gru_conf else "GRU"
            
            if best_word:
                st.markdown(f"""
                <div class="result-card">
                    <p style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.5rem;">Recommended Output (via {winning_model})</p>
                    <p style="font-size: 1.1rem; color: #374151;">
                        {input_text} <span style="color: #2563eb; font-weight: 600; background-color: #eff6ff; padding: 0.2rem 0.5rem; border-radius: 4px;">{best_word}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Both models failed to predict the next sequence. Out-of-vocabulary tokens may be present.")