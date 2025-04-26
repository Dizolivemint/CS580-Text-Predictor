# 🤖✨ AI Text Predictor: LLM-Augmented Data Edition

Welcome to the **AI Text Predictor Project**! This workflow uses local LLMs (with Ollama 🦙) to automatically generate next-word or next-phrase predictions from real movie dialogue data. You’ll be building a *smarter* and more *context-aware* text prediction dataset—perfect for training modern ML models!

---

## 🗂️ Project Structure

ai-text-predictor-llm/
│
├── .venv/                   # Your local Python virtual environment (add to .gitignore)
├── data/
│   ├── train-00000-of-00001-a19c10f9666706bb.parquet  # Original dataset
│   └── context_target_pairs.csv       # Processed context/target data
├── scripts/
│   └── batch_ollama_prompt.py         # Resumable batch LLM prompting script
├── outputs/
│   └── llm_augmented_dataset.csv      # The growing, augmented dataset
├── README.md
├── requirements.txt
└── .gitignore


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

