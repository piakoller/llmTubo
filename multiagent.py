# multiagent.py
import requests
import os
import urllib.parse
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from typing import Union

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- Configuration ---
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_PAGE_SIZE = 10
REQUESTS_TIMEOUT = 30 # Timeout for API calls in seconds
MAX_LOCATIONS_TO_DISPLAY = 3
GEOLOCATOR = Nominatim(user_agent="llm_tumorboard_app") 
GEOCODE_CACHE = {}  # Cache for geocoding results

# --- Logging ---
# Logger can be configured externally, but setting a default here is helpful
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s")

def get_geopoint(location_string: str) -> Union[tuple[float, float], None]:
    """Geocodes a location string to latitude and longitude using Nominatim."""
    if not location_string or not location_string.strip():
        return None
    normalized_location = location_string.strip().lower()
    if normalized_location in GEOCODE_CACHE:
        return GEOCODE_CACHE[normalized_location]

    logger.info(f"Geocoding location: '{location_string}'")
    try:
        # Increase timeout for geocoding as it's an external service call
        location = GEOLOCATOR.geocode(location_string, timeout=10)
        if location:
            geopoint = (location.latitude, location.longitude)
            GEOCODE_CACHE[normalized_location] = geopoint
            logger.info(f"Geocoded '{location_string}' to {geopoint}")
            return geopoint
        else:
            logger.warning(f"Could not geocode location: '{location_string}'")
            GEOCODE_CACHE[normalized_location] = None # Cache failed lookups too
            return None
    except Exception as e:
        logger.error(f"Geocoding failed for '{location_string}': {e}")
        GEOCODE_CACHE[normalized_location] = None # Cache failed lookups too
        return None

# --- Agent Runner ---
class AgentRunner:
    """Handles the execution of an agent, capturing results, runtime, and exceptions."""
    def __init__(self, agent: 'Agent', name: str, context: str, extra_args: dict = None):
        self.agent = agent
        self.name = name
        self.context = context
        self.extra_args = extra_args or {}
        self.result = None
        self.runtime = None
        self.exception = None

    def run(self):
        """Executes the agent's respond method and records metrics."""
        logger.info(f"Agent '{self.name}' started.")
        start_time = time.perf_counter()
        try:
            self.result = self.agent.respond(self.context)
        except Exception as e:
            logger.error(f"Agent '{self.name}' encountered an error: {e}", exc_info=True)
            self.exception = e
        finally:
            self.runtime = time.perf_counter() - start_time
            status = "failed" if self.exception else "completed"
            logger.info(f"Agent '{self.name}' {status} in {self.runtime:.2f}s.")


# --- Base Agent Class ---
class Agent(ABC):
    """Abstract base class for all agents."""
    def __init__(self, name: str, role_description: str, llm: OllamaLLM):
        self.name = name
        self.role_description = role_description
        if llm is None:
            raise ValueError("LLM instance must be provided to the agent.")
        self.llm = llm
        logger.info(f"Agent '{self.name}' initialized.")

    @abstractmethod
    def respond(self, context: str) -> any:
        """
        Process the given context and return a response.
        The response type depends on the specific agent.
        """
        pass

    def _invoke_llm(self, prompt_template: str, context_vars: dict) -> str:
        """Helper method to invoke the LLM with a specific template."""
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        try:
            response = chain.invoke(context_vars)
            return response
        except Exception as e:
            logger.error(f"LLM invocation failed for agent {self.name}: {e}", exc_info=True)
            # Re-raise or return a specific error message
            raise RuntimeError(f"LLM call failed for {self.name}") from e


class DiagnostikAgent(Agent):
    """Analyzes diagnosis, symptoms, and clinical reports."""
    def __init__(self, llm: OllamaLLM):
        super().__init__("Diagnostik", "Analysiere Diagnose, Symptome und klinische Berichte und fasse die wichtigsten Punkte zusammen.", llm)

    def respond(self, context: str) -> str:
        """Uses LLM to analyze the provided context."""
        template = """
        Du bist ein medizinischer Experte für Diagnostik. Deine Rolle ist: {role}
        Analysiere den folgenden Patientenkontext und fasse die relevanten diagnostischen Informationen, das Stadium und wichtige klinische Punkte klar und prägnant zusammen.

        Kontext:
        ---
        {context}
        ---

        Zusammenfassung der Diagnostik:
        """
        return self._invoke_llm(template, {"role": self.role_description, "context": context})


class StudienAgent(Agent):
    """Searches for relevant clinical trials on clinicaltrials.gov."""
    def __init__(self, llm: OllamaLLM, location: str = None):
        # This agent currently doesn't use the LLM directly, but keeps the interface consistent
        super().__init__(
            "Studienrecherche",
            "Filtere relevante klinische Studien von clinicaltrials.gov basierend auf Diagnose, Symptomen, Stadium und Ort.",
            llm
        )
        self.location_string = location
        logger.info(f"StudienAgent initialized with location: '{self.location_string}'")

        if self.location_string:
            # Geocode the user's input location once during initialization
            self.location_geopoint = get_geopoint(self.location_string)
        if not self.location_geopoint:
            logger.warning(f"Could not geocode provided location: '{self.location_string}'. Distance calculation will not be possible.")
        logger.info(f"StudienAgent initialized with location string: '{self.location_string}'. Geopoint: {self.location_geopoint}")

    def _search_clinical_trials(self, diagnosis: str) -> List[Dict[str, Any]]:
        """
        Performs the search on clinicaltrials.gov API (v2 endpoint),
        calculates distances, and extracts location data.
        """
        if not diagnosis:
             logger.warning("Skipping clinical trial search: Diagnosis is missing.")
             return []

        search_terms = f"{diagnosis}"
        query = urllib.parse.quote_plus(search_terms)

        # Build base URL
        # url = f"{CLINICAL_TRIALS_API_URL}?query.term={query}&pageSize={CLINICAL_TRIALS_PAGE_SIZE}"
        url = f"{CLINICAL_TRIALS_API_URL}?query.term={query}"

        # Use query.locn for filtering by location string BEFORE getting results
        # This is still useful for limiting the initial search pool to relevant areas.
        # Distance calculation refines this *after* fetching.
        if self.location_string:
            encoded_location = urllib.parse.quote_plus(self.location_string.strip())
            url += f"&query.locn={encoded_location}"
            logger.info(f"Adding location string filter to URL: '{self.location_string}' using query.locn")

        logger.info(f"Searching clinical trials with URL: {url}")

        try:
            response = requests.get(url, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            studies = data.get("studies", [])
            processed_studies = [] # List to hold studies with added distance info

            for study in studies:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                desc_module = protocol.get("descriptionModule", {})
                locations_module = protocol.get("contactsLocationsModule", {})
                moreInfoModule = protocol.get("moreInfoModule", {})

                title = id_module.get("officialTitle", "N/A")
                nct_id = id_module.get("nctId", "NCT ID nicht verfügbar")
                status = status_module.get("overallStatus", "Status unbekannt")
                summary = desc_module.get("briefSummary", "Keine Zusammenfassung verfügbar.")

                locations_list_with_distances = []
                api_locations = locations_module.get("locations", [])
                
                min_distance_km = float('inf') # Track minimum distance for this study
                contact = moreInfoModule.get("contact", {})

                # Process each location within the study
                for loc in api_locations:
                    facility = loc.get("facility", "N/A")
                    city = loc.get("city", "N/A")
                    state = loc.get("state", "") # State is optional
                    country = loc.get("country", "N/A")
                    geo_point_data = loc.get("geoPoint") # Get geopoint data

                    location_str_parts = [facility] # Start with facility
                    if city and city != "N/A":
                         location_str_parts.append(city)
                    if state and state != "": # Only add state if it exists
                         location_str_parts.append(state)
                    if country and country != "N/A":
                         location_str_parts.append(country)

                    # Join parts, removing any N/A or empty strings and duplicates
                    location_parts_cleaned = [p for p in location_str_parts if p not in ["N/A", ""] and p is not None]
                    # Remove potential duplicates like "Bern, Bern, Switzerland" if state=city
                    location_parts_unique = []
                    for p in location_parts_cleaned:
                        if p not in location_parts_unique:
                            location_parts_unique.append(p)
                    location_str = ", ".join(location_parts_unique)
                    if not location_str: location_str = "Keine Ortsangaben verfügbar" # Fallback if all parts are missing

                    distance_km = None # Distance for this specific location

                    # Calculate distance if source location was geocoded and study location has geopoint
                    if self.location_geopoint and geo_point_data:
                        try:
                            study_lat = geo_point_data.get('lat')
                            study_lon = geo_point_data.get('lon')
                            if study_lat is not None and study_lon is not None:
                                study_geopoint = (study_lat, study_lon)
                                distance_km = geodesic(self.location_geopoint, study_geopoint).km
                                min_distance_km = min(min_distance_km, distance_km) # Update min distance for the study
                        except Exception as e:
                            logger.debug(f"Could not calculate geodesic distance for study {nct_id}, location {location_str}: {e}")
                            # distance_km remains None

                    locations_list_with_distances.append({
                        "name": location_str,
                        "distance_km": distance_km # Store distance for this specific location
                    })

                # Handle case where min_distance_km is still infinity (no locations or geocoding failed)
                final_min_distance = min_distance_km if min_distance_km != float('inf') else None

                processed_studies.append({
                    "title": title,
                    "nct_id": nct_id,
                    "status": status,
                    "summary": summary,
                    "locations": locations_list_with_distances, # List now contains dicts with distance
                    "min_distance_km": final_min_distance # Store the minimum distance for sorting
                })

            logger.info(f"Found {len(processed_studies)} studies after initial API filter.")

            # --- Sort studies by minimum distance ---
            # Sorts studies where min_distance_km is not None first, then by distance ascending
            # Studies with no geocodable locations will appear at the end
            sorted_studies = sorted(processed_studies, key=lambda x: x.get('min_distance_km') if x.get('min_distance_km') is not None else float('inf'))
            logger.info(f"Sorted {len(sorted_studies)} studies by minimum distance.")
            # --- End Sort ---

            return sorted_studies

        except requests.exceptions.Timeout:
             logger.error(f"API request timed out for clinical trials search: {url}")
             return []
        except requests.exceptions.RequestException as e:
            logger.error(f"API error during clinical trials search ({url}): {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during clinical trials processing: {e}", exc_info=True)
            return []

    def _parse_context(self, diagnosis_context: str) -> str: # Return type hint is str now
        """Extracts diagnosis from the context string for search terms."""
        diagnosis = ""
        context_lines = diagnosis_context.strip().split('\n')
        for line in context_lines:
            line = line.strip()
            if line.lower().startswith("diagnose:"):
                 # Assuming format "Diagnose: CODE (TEXT)"
                 parts = line.split(":", 1)[1].strip().split("(", 1)
                 if len(parts) > 1:
                    main_diagnosis_text = parts[1].split(")", 1)[0].strip()
                    if main_diagnosis_text:
                         diagnosis = main_diagnosis_text
                    else: # Fallback if parentheses are empty
                         diagnosis = parts[0].strip()
                 else:
                    # If no text in parentheses, just use the code/value after "Diagnose:"
                    diagnosis = line.split(":", 1)[1].strip()
                 break
        return diagnosis

    def respond(self, diagnosis_context: str) -> list[dict]:
        """Parses context, searches trials using the stored location, and returns results as list of dicts."""
        diagnosis_search_term = self._parse_context(diagnosis_context)
        trials = self._search_clinical_trials(diagnosis_search_term)

        if not trials:
            logger.info("No suitable clinical trials found or search failed.")
            return []

        # Return list of dictionaries directly
        return trials

class TherapieAgent(Agent):
    """Generates a therapy recommendation based on provided information."""
    def __init__(self, llm: OllamaLLM, guideline_provider: str):
        super().__init__(
            "Therapieempfehlung",
            "Erstelle eine detaillierte, leitlinienbasierte Therapieempfehlung basierend auf der Diagnoseanalyse.",
             llm
         )
        # Store the guideline provider
        self.guideline_provider = guideline_provider
        logger.info(f"TherapieAgent initialized with guideline: {self.guideline_provider}")

    def respond(self, context: str) -> str:
        """Uses LLM to generate therapy recommendations."""
        template = """
            # Therapieempfehlung für Tumorboard

            Du bist ein erfahrener Onkologie-Experte, spezialisiert auf Therapieplanung. Deine Aufgabe ist es, basierend auf der bereitgestellten diagnostischen Zusammenfassung und unter strikter Berücksichtigung der angegebenen Leitlinie, eine präzise und begründete Therapieempfehlung für den besprochenen Patienten zu erstellen.

            **Deine Rolle:** {role}

            **Leitlinie für diese Empfehlung:** {guideline_provider}

            **Patienten-Kontext und Diagnostische Zusammenfassung:**
            ---
            {context}
            ---

            **Anweisungen für die Erstellung der Therapieempfehlung:**
            1.  Analysiere die bereitgestellten Informationen im Abschnitt "Patienten-Kontext und Diagnostische Zusammenfassung" sorgfältig.
            2.  Berücksichtige **ausschließlich** die angegebene Leitlinie: **{guideline_provider}**. Ignoriere andere Leitlinien oder allgemeine Informationen, es sei denn, sie sind im Kontext als relevant aufgeführt.
            3.  Formuliere **EINE EINZIGE, KONKRETE THERAPIETHERAPIE**, die am besten zu den Fakten im Kontext und der angegebenen Leitlinie passt.
            4.  Sollte im Kontext bereits eine mögliche Therapie vorgeschlagen sein, bewerte diese kritisch im Licht der Leitlinie und integriere oder ersetze sie durch die am besten begründete Empfehlung.
            5.  Gib eine klare und prägnante Begründung für die empfohlene Therapie. Die Begründung muss sich explizit auf die relevanten Punkte aus dem Patienten-Kontext (Diagnose, Stadium, relevante Vorerkrankungen etc.) und die Empfehlungen der Leitlinie beziehen.
            6.  Sollten die im Kontext enthaltenen Informationen für eine definitive Therapieempfehlung unzureichend sein (z.B. fehlendes Staging, unklare Histologie), gib dies an und schlage notwendige weitere diagnostische Schritte vor.
            7.  Gib **ausschließlich** die Therapieempfehlung und die Begründung aus. Verwende dabei die unten angegebene Struktur der Überschriften. Füge keine zusätzlichen einleitenden oder abschliessenden Sätze hinzu, die nicht Teil der Empfehlung oder Begründung sind.

            **Deine Antwort (Im folgenden Markdown-Format):**

            **Therapieempfehlung:**
            [Hier deine einzige, konkrete Therapieempfehlung einfügen. Beginne direkt mit der Empfehlung.]

            **Begründung:**
            [Hier deine detaillierte Begründung einfügen. Beziehe dich auf den Kontext und die Leitlinie.]
            """
        return self._invoke_llm(template, {"role": self.role_description, "guideline_provider": self.guideline_provider, "context": context})

class ReportAgent:
    """Generates and saves a structured medical report."""
    def __init__(self, llm: OllamaLLM, output_dir: str = "generated_report", file_type: str = "md"):
        if llm is None:
             raise ValueError("LLM instance must be provided to the ReportAgent.")
        self.llm = llm
        self.output_dir = output_dir
        # Ensure file type is supported (currently only md)
        self.file_type = file_type.lower()
        if self.file_type not in ["md", "markdown"]:
            logger.warning(f"Unsupported file type '{file_type}'. Defaulting to 'md'.")
            self.file_type = "md"
        logger.info(f"ReportAgent initialized. Output directory: '{output_dir}', File type: '{self.file_type}'")


    def generate_report_text(self, context: str, patient_data: dict, board_date: str) -> str:
        """Uses LLM to structure the context into a report format, incorporating patient data."""
        logger.info("Generating report text...")

        # Prepare patient info string for the prompt
        # Adjust keys based on your actual patient_data dictionary
        patient_info_str = f"""
            Patient: {patient_data.get('last_name', 'Nachname')}, {patient_data.get('first_name', 'Vorname')}
            Geburtsdatum: {patient_data.get('dob', 'XX.XX.XXXX')}
            PID: {patient_data.get('pid', 'XXXXXXXX')}
            """

        # The main diagnosis might be in patient_data or context, adjust as needed
        main_diagnosis_str = patient_data.get('main_diagnosis_text', 'Hauptdiagnose nicht spezifiziert')

        prompt = PromptTemplate(
            input_variables=["patient_info", "board_date", "main_diagnosis", "context"],
            template="""
            Du bist ein medizinischer Sekretär/Assistent, der Tumorboard-Protokolle im Stil des Inselspitals Bern (Schweiz) verfasst.
            Erstelle ein Protokoll für das Endokrine Tumorboard vom {board_date}.

            **Patientendaten:**
            {patient_info}

            **Hauptdiagnose (basierend auf vorliegenden Informationen):**
            1. {main_diagnosis}

            Der folgende `{context}` enthält eine Zusammenfassung der aktuellen diagnostischen Situation durch einen Experten und die daraus resultierende aktuelle Therapieempfehlung des Tumorboards.

            **Eingabeinformationen (Aktuelle Beurteilung & Empfehlung):**
            ---
            {context}
            ---

            **Deine Aufgabe:**
            Formatiere **nur** die Informationen aus dem `{context}` in ein medizinisches Protokoll im Markdown-Format.
            Orientiere dich am Stil des Inselspitals, insbesondere bei den Abschnitten 'Beurteilung' und 'Empfehlungen'.
            Fasse die diagnostische Einschätzung aus dem `{context}` unter der Überschrift '## Beurteilung' zusammen.
            Liste die aktuelle Therapieempfehlung(en) aus dem `{context}` unter der Überschrift '## Empfehlungen' auf, idealerweise als Bullet Points (`*` oder `•`).

            **WICHTIG:**
            *   Füge **keine** historischen Daten (frühere Scans, Behandlungen, Empfehlungen, Labore, Verläufe) hinzu, die nicht explizit im `{context}` stehen.
            *   Erfinde **keine** Patientendaten, Adressen, Board-Teilnehmer oder Abteilungsleiter.
            *   Generiere **keinen** Briefkopf oder Fusszeile.
            *   Konzentriere dich darauf, die *aktuelle Beurteilung* und die *neuen Empfehlungen* basierend auf dem `{context}` klar darzustellen.

            **Protokoll (Markdown):**

            ## Beurteilung
            [Hier die Zusammenfassung der diagnostischen Einschätzung aus dem context einfügen]

            ## Empfehlungen
            [Hier die aktuelle(n) Therapieempfehlung(en) aus dem context als Bullet Points auflisten]

            """
        )
        chain = prompt | self.llm
        try:
            report_text = chain.invoke({
                "patient_info": patient_info_str,
                "board_date": board_date,
                "main_diagnosis": main_diagnosis_str,
                "context": context
                })
            logger.info("Report text generated successfully.")
            # Prepend a basic header to the LLM output
            header = f"# Endokrines Tumorboard vom {board_date}\n\n"
            full_report = header + patient_info_str + f"\n## Diagnosen\n1. {main_diagnosis_str}\n\n" + report_text
            return full_report
        except Exception as e:
            logger.error(f"LLM invocation failed during report generation: {e}", exc_info=True)
            raise RuntimeError("LLM call failed during report generation") from e

    def save_report(self, report_text: str, filename_base: str) -> str:
        """Saves the generated report text to a file."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f"{filename_base}.{self.file_type}")
        logger.info(f"Attempting to save report to: {filepath}")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Report successfully saved to {filepath}")
            return filepath
        except OSError as e:
            logger.error(f"Failed to save report to {filepath}: {e}", exc_info=True)
            raise RuntimeError(f"Error saving report file: {e}") from e
        except Exception as e: # Catch any other unexpected errors during file writing
             logger.error(f"An unexpected error occurred while saving the report to {filepath}: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error saving report file: {e}") from e