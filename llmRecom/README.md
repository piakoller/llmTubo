# Multi-Agent Tumor Board Therapy Recommendation System 🧬🔬📄

This project implements a multi-agent system using Streamlit and Langchain with Ollama to simulate a tumor board. It takes patient data, analyzes it through specialized agents (Diagnostik, Studien, Therapie), and generates a therapy recommendation, relevant clinical studies, and a final report.

---

## ✨ Core Functionality

* **Patient Data Input:** Loads anonymized patient data from an Excel file.
* **Interactive UI:** Allows selection of a patient, relevant medical guidelines (ESMO, Onkopedia, S3), and a location for clinical trial searches.
* **Multi-Agent Workflow:**
  * **Diagnostik Agent:** Summarizes the patient's diagnostic information.
  * **Studien Agent:** Searches ClinicalTrials.gov for relevant studies based on diagnosis and location, providing distance-sorted results.
  * **Therapie Agent:** Generates a therapy recommendation based on the diagnostic summary and selected guidelines.
  * **Report Agent:** Compiles the findings into a structured medical report.
* **LLM Integration:** Leverages a local LLM (e.g., Qwen3:32b via Ollama) for natural language understanding and generation tasks performed by the agents.
* **Modularity:** Code is structured into UI, core logic, agents, and services for better maintainability.

---

## ✨ Features

* **Streamlit Interface:** Easy-to-use web application for interaction.
* **Dynamic Patient Selection:** Load and choose from multiple patient profiles.
* **Guideline-Specific Recommendations:** Therapy suggestions consider user-selected medical guidelines.
* **Localized Clinical Trial Search:** Find studies near a specified location with distance calculation.
* **Parallel Agent Execution:** Diagnostik and Studien agents run in parallel for efficiency.
* **Automated Report Generation:** Produces a downloadable Markdown report.
* **LLM-Powered Analysis:** Utilizes large language models for intelligent text processing.
* **Configurable:** Key parameters (LLM model, file paths) managed via `config.py`.
* **Structured Logging:** Detailed logging for easier debugging and monitoring.

---

## 🔧 Tech Stack

* **Frontend:** Streamlit
* **Backend/Orchestration:** Python
* **LLM Framework:** Langchain
* **LLM Provider:** Ollama (for local LLM hosting, e.g., Qwen3:32b, Llama3, etc.)
* **Geocoding:** `geopy` (Nominatim)
* **Clinical Trials API:** ClinicalTrials.gov
* **Data Handling:** Pandas

---

## 🏛️ Architecture Overview

The system is designed with a modular architecture:

1. **UI Layer (`app.py`, `ui/`)**: Handles user interaction, data input, and results display using Streamlit components.
2. **Core Logic (`core/agent_manager.py`)**: Orchestrates the multi-agent workflow, initializes components, prepares contexts, and manages agent execution.
3. **Agent Layer (`agents/`)**: Contains the definitions for specialized agents (`DiagnostikAgent`, `StudienAgent`, `TherapieAgent`, `ReportAgent`) and the base `Agent` class with `AgentRunner`.
4. **Services Layer (`services/`)**: Provides utilities like geocoding (`geocoding_service.py`).
5. **Data & Configuration (`data_loader.py`, `config.py`, `utils/`)**: Manages data loading, application settings, and logging setup.
6. **External Services**: Ollama (LLM), ClinicalTrials.gov API, Geocoding API.

---

## 🚀 How to Use

### 1. **Setup the Environment**
   - Clone the repository:
     ```bash
     git clone https://github.com/your-repo/llmTubo.git
     cd llmTubo
     ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

### 2. **Configure the Application**
   - Edit the [config.py](http://_vscodecontentref_/0) file to set up:
     - LLM model and temperature.
     - File paths for patient data and report output.
     - API keys for external services (if required).

### 3. **Prepare Patient Data**
   - Ensure patient data is available in the specified Excel format.
   - Place the file in the appropriate directory as configured in [config.py](http://_vscodecontentref_/1).

### 4. **Run the Application**
   - Start the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - Open the provided URL in your browser (usually `http://localhost:8501`).

### 5. **Interact with the Application**
   - Select a patient from the dropdown menu.
   - Choose medical guidelines and a location for clinical trial searches.
   - Click "Run Workflow" to execute the multi-agent system.
   - Download the generated report.

---

## 🛠️ Troubleshooting

* **Dependency Issues:** Ensure all dependencies are installed using `pip install -r requirements.txt`.
* **LLM Errors:** Verify that the LLM model is correctly configured in [config.py](http://_vscodecontentref_/2) and accessible via Ollama.
* **Geocoding Issues:** Check your internet connection and ensure the `geopy` library is installed.
* **Debugging:** Review the logs in the console or log files for detailed error messages.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

Let me know if you need further improvements or additional sections! 😊