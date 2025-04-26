# 🤖✨ AI Text Predictor: LLM-Augmented Data Edition

Welcome to the **AI Text Predictor Project**! This workflow uses local LLMs (with Ollama 🦙) to automatically generate next-word or next-phrase predictions from real movie dialogue data. You’ll be building a *smarter* and more *context-aware* text prediction dataset—perfect for training modern ML models!

## ⚠️ Notes & Important Changes
### 🧪 PyTorch Nightly with CUDA 12.8
This project utilizes PyTorch 2.8.0.dev20250425+cu128, installed from the nightly CUDA 12.8 builds. This ensures compatibility with NVIDIA GeForce RTX 50 Series GPUs (Blackwell architecture, compute capability sm_120).​

### 🐍 Python 3.11 Compatibility

The project is developed using Python 3.11. Ensure that your virtual environment is set up with this version to maintain compatibility.

---

## 🗂️ Project Structure

- `.venv/`  
  Python virtual environment (not tracked by git)
- `data/`
  - `train-00000-of-00001-a19c10f9666706bb.parquet`  
    Original dataset
  - `context_target_pairs.csv`  
    Processed context/target data
- `scripts/`
  - `batch_ollama_prompt.py`  
    Resumable LLM batch prompting script
- `outputs/`
  - `llm_augmented_dataset.csv`  
    LLM-augmented predictions
- `README.md`
- `requirements.txt`
- `.gitignore`


---

## 🚀 Setup (with .venv)

1. **Clone the repo and enter the folder:**
    ```bash
    git clone <your-repo-url>
    cd ai-text-predictor-llm
    ```

2. **Create your virtual environment (using `.venv`):**
    ```bash
    python -m venv .venv
    ```

3. **Activate the environment:**
    - **On macOS/Linux:**
      ```bash
      source .venv/bin/activate
      ```
    - **On Windows:**
      ```cmd
      .venv\Scripts\activate
      ```

4. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🤖 LLM Batch Prompting with Ollama (Resumable!)

- Use `scripts/batch_ollama_prompt.py` to automate sending context prompts to your local Ollama LLM.
- The script:
    - Reads from your processed dataset
    - Sends each context to Ollama
    - Saves LLM responses as predictions
    - **Checkpoints your progress automatically!** So you can pause and resume anytime.

---

## 🦙 Requirements

- **Ollama** running locally ([see their docs](https://ollama.com/))
- **Python 3.8+**
- **Key packages:** `pandas`, `requests`, `pyarrow`

---

## 💾 Checkpoints = Peace of Mind

Don’t worry if your run gets interrupted—the script saves after every batch, so you’ll never lose progress! 🔁

---

## 📈 Next Steps

- Preprocess your dataset (see `data/`)
- Run the batch prompting script
- Move on to model training or demo apps (Gradio/Streamlit!)

---

**Have fun—and may your predictions always be spot-on!** 😎✨

---

