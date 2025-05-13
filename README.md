# Multi-Agent Tumor Board Therapy Recommendation System 🧬🔬📄

This project implements a multi-agent system using Streamlit and Langchain with Ollama to simulate a tumor board. It takes patient data, analyzes it through specialized agents (Diagnostik, Studien, Therapie), and generates a therapy recommendation, relevant clinical studies, and a final report.

**Core Functionality:**

*   **Patient Data Input:** Loads anonymized patient data from an Excel file.
*   **Interactive UI:** Allows selection of a patient, relevant medical guidelines (ESMO, Onkopedia, S3), and a location for clinical trial searches.
*   **Multi-Agent Workflow:**
    *   **Diagnostik Agent:** Summarizes the patient's diagnostic information.
    *   **Studien Agent:** Searches ClinicalTrials.gov for relevant studies based on diagnosis and location, providing distance-sorted results.
    *   **Therapie Agent:** Generates a therapy recommendation based on the diagnostic summary and selected guidelines.
    *   **Report Agent:** Compiles the findings into a structured medical report.
*   **LLM Integration:** Leverages a local LLM (e.g., Qwen3:32b via Ollama) for natural language understanding and generation tasks performed by the agents.
*   **Modularity:** Code is structured into UI, core logic, agents, and services for better maintainability.

---

## ✨ Features

*   **Streamlit Interface:** Easy-to-use web application for interaction.
*   **Dynamic Patient Selection:** Load and choose from multiple patient profiles.
*   **Guideline-Specific Recommendations:** Therapy suggestions consider user-selected medical guidelines.
*   **Localized Clinical Trial Search:** Find studies near a specified location with distance calculation.
*   **Parallel Agent Execution:** Diagnostik and Studien agents run in parallel for efficiency.
*   **Automated Report Generation:** Produces a downloadable Markdown report.
*   **LLM-Powered Analysis:** Utilizes large language models for intelligent text processing.
*   **Configurable:** Key parameters (LLM model, file paths) managed via `config.py`.
*   **Structured Logging:** Detailed logging for easier debugging and monitoring.

---

## 🔧 Tech Stack

*   **Frontend:** Streamlit
*   **Backend/Orchestration:** Python
*   **LLM Framework:** Langchain
*   **LLM Provider:** Ollama (for local LLM hosting, e.g., Qwen3:32b, Llama3, etc.)
*   **Geocoding:** `geopy` (Nominatim)
*   **Clinical Trials API:** ClinicalTrials.gov
*   **Data Handling:** Pandas

---

## 🏛️ Architecture Overview

The system is designed with a modular architecture:

1.  **UI Layer (`app.py`, `ui/`)**: Handles user interaction, data input, and results display using Streamlit components.
2.  **Core Logic (`core/agent_manager.py`)**: Orchestrates the multi-agent workflow, initializes components, prepares contexts, and manages agent execution.
3.  **Agent Layer (`agents/`)**: Contains the definitions for specialized agents (`DiagnostikAgent`, `StudienAgent`, `TherapieAgent`, `ReportAgent`) and the base `Agent` class with `AgentRunner`.
4.  **Services Layer (`services/`)**: Provides utilities like geocoding (`geocoding_service.py`).
5.  **Data & Configuration (`data_loader.py`, `config.py`, `utils/`)**: Manages data loading, application settings, and logging setup.
6.  **External Services**: Ollama (LLM), ClinicalTrials.gov API, Geocoding API.

*(You can optionally embed the Mermaid diagram here if your GitHub Markdown renderer supports it, or link to an image of the diagram)*

```mermaid
graph TD
    subgraph UserInterface [Streamlit App]
        direction LR
        AppPy["app.py (Main UI)"]
        Sidebar["ui/sidebar.py"]
        PatientForm["ui/patient_form.py"]
        ResultsDisplay["ui/results_display.py"]
    end
    subgraph CoreLogic; AgentManager["core/agent_manager.py"]; end
    subgraph Agents
        direction TB
        BaseAgent["agents/base_agent.py"]
        DiagnostikAgent["agents/DiagnostikAgent"]
        StudienAgent["agents/StudienAgent"]
        TherapieAgent["agents/TherapieAgent"]
        ReportAgent["agents/ReportAgent"]
    end
    subgraph Services; GeoService["services/geocoding_service.py"]; end
    subgraph DataAndConfig
        direction TB
        ConfigPy["config.py"]
        DataLoader["data_loader.py"]
        ExcelFile["Patient Data (Excel)"]
        LoggingSetup["utils/logging_setup.py"]
    end
    subgraph ExternalServices
        Ollama["Ollama LLM"]
        ClinicalTrialsAPI["ClinicalTrials.gov API"]
        GeocodingAPI["Geocoding API"]
    end
    AppPy --> AgentManager; AgentManager --> AppPy;
    AppPy --> Sidebar; AppPy --> PatientForm; AppPy --> ResultsDisplay;
    ResultsDisplay --> ReportAgent;
    AppPy --> DataLoader; DataLoader --> ExcelFile; DataLoader --> ConfigPy;
    AppPy --> LoggingSetup; LoggingSetup --> ConfigPy;
    AgentManager --> BaseAgent; AgentManager --> DiagnostikAgent; AgentManager --> StudienAgent;
    AgentManager --> TherapieAgent; AgentManager --> ReportAgent;
    AgentManager --> ConfigPy; AgentManager --> GeoService;
    DiagnostikAgent --> BaseAgent; StudienAgent --> BaseAgent; TherapieAgent --> BaseAgent; ReportAgent --> BaseAgent;
    DiagnostikAgent --> Ollama; TherapieAgent --> Ollama; ReportAgent --> Ollama;
    StudienAgent --> ClinicalTrialsAPI; StudienAgent --> GeoService; GeoService --> GeocodingAPI;
    classDef ui fill:#D6EAF8,stroke:#5DADE2; classDef core fill:#D5F5E3,stroke:#58D68D; classDef agent fill:#FADBD8,stroke:#EC7063; classDef service fill:#E8DAEF,stroke:#A569BD; classDef data fill:#FCF3CF,stroke:#F4D03F; classDef external fill:#EAEDED,stroke:#AEB6BF;
    class AppPy,Sidebar,PatientForm,ResultsDisplay ui; class AgentManager core; class BaseAgent,DiagnostikAgent,StudienAgent,TherapieAgent,ReportAgent agent; class GeoService service; class ConfigPy,DataLoader,ExcelFile,LoggingSetup data; class Ollama,ClinicalTrialsAPI,GeocodingAPI external;