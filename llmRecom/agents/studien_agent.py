# agents/studien_agent.py
import logging
import urllib.parse
import requests
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM

from .base_agent import Agent # Relative import
from services.geocoding_service import get_geopoint, calculate_distance # Service import
import config # Global config

logger = logging.getLogger(__name__)

class StudienAgent(Agent):
    def __init__(self, llm: OllamaLLM, search_location_str: str | None = None):
        super().__init__(
            name="Studienrecherche",
            role_description="Filtere relevante klinische Studien von clinicaltrials.gov.",
            llm=llm # LLM not directly used for search, but part of base class
        )
        self.search_location_str = search_location_str
        self.search_location_geopoint = get_geopoint(search_location_str) if search_location_str else None
        logger.info(f"StudienAgent initialized. Search location: '{self.search_location_str}', Geopoint: {self.search_location_geopoint}")

    def _fetch_studies_from_api(self, diagnosis_search_term: str) -> List[Dict[str, Any]]:
        """Fetches raw study data from ClinicalTrials.gov API."""
        if not diagnosis_search_term:
            logger.warning("Skipping API call: Diagnosis search term is empty.")
            return []

        query = urllib.parse.quote_plus(diagnosis_search_term)
        url = f"{config.CLINICAL_TRIALS_API_URL}?query.term={query}&pageSize={config.CLINICAL_TRIALS_PAGE_SIZE}"

        if self.search_location_str:
            encoded_location = urllib.parse.quote_plus(self.search_location_str.strip())
            url += f"&query.locn={encoded_location}"
            logger.info(f"Adding location filter to API URL: '{self.search_location_str}'")

        logger.info(f"Requesting studies from API: {url}")
        try:
            response = requests.get(url, timeout=config.REQUESTS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            studies_data = data.get("studies", [])
            logger.info(f"API returned {len(studies_data)} studies for term '{diagnosis_search_term}'.")
            return studies_data
        except requests.exceptions.Timeout:
            logger.error(f"API request timed out for URL: {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"API error for URL {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching studies: {e}", exc_info=True)
        return []

    def _process_and_enrich_study(self, study_data: Dict[str, Any]) -> Dict[str, Any] | None:
        """Processes a single raw study, extracts fields, and calculates distances for its locations."""
        try:
            protocol = study_data.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            desc_module = protocol.get("descriptionModule", {})
            locations_module = protocol.get("contactsLocationsModule", {})

            study_info = {
                "title": id_module.get("officialTitle") or id_module.get("briefTitle", "N/A"),
                "nct_id": id_module.get("nctId", "N/A"),
                "status": status_module.get("overallStatus", "Unbekannt"),
                "summary": desc_module.get("briefSummary", "Keine Zusammenfassung."),
                "locations": [],
                "min_distance_km": float('inf')
            }

            api_locations = locations_module.get("locations", [])
            for loc_data in api_locations:
                facility = loc_data.get("facility", "N/A")
                city = loc_data.get("city")
                state = loc_data.get("state")
                country = loc_data.get("country")
                geo_point_data = loc_data.get("geoPoint") # {"lat": ..., "lon": ...}

                loc_parts = [p for p in [facility, city, state, country] if p and p != "N/A"]
                # Remove duplicates while preserving order for readability
                unique_loc_parts = []
                for part in loc_parts:
                    if part not in unique_loc_parts:
                        unique_loc_parts.append(part)
                loc_name = ", ".join(unique_loc_parts) if unique_loc_parts else "Standort nicht spezifiziert"

                distance_km = None
                if self.search_location_geopoint and geo_point_data and \
                   geo_point_data.get('lat') is not None and geo_point_data.get('lon') is not None:
                    try:
                        study_loc_geopoint = (float(geo_point_data['lat']), float(geo_point_data['lon']))
                        dist = calculate_distance(self.search_location_geopoint, study_loc_geopoint)
                        if dist is not None:
                            distance_km = dist
                            study_info["min_distance_km"] = min(study_info["min_distance_km"], dist)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error processing geo data for study {study_info['nct_id']}, loc {loc_name}: {e}")
                
                study_info["locations"].append({"name": loc_name, "distance_km": distance_km})
            
            if study_info["min_distance_km"] == float('inf'):
                study_info["min_distance_km"] = None # No valid distance found
            
            # Sort individual locations by distance for this study
            study_info["locations"] = sorted(
                [loc for loc in study_info["locations"] if loc.get("distance_km") is not None], # only sortable ones
                key=lambda x: x["distance_km"] 
            ) + [loc for loc in study_info["locations"] if loc.get("distance_km") is None] # add non-sortable at end


            return study_info
        except Exception as e:
            nct_id = study_data.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "UnknownNCTID")
            logger.error(f"Error processing study {nct_id}: {e}", exc_info=True)
            return None


    def _sort_studies_by_overall_distance(self, studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sorts studies by their 'min_distance_km', None distances last."""
        if not self.search_location_geopoint:
            return studies # No reference point for sorting
            
        return sorted(
            studies,
            key=lambda s: (s.get('min_distance_km') is None, s.get('min_distance_km', float('inf')))
        )

    def respond(self, diagnosis_search_term: str, attachments: list[str] | None = None) -> list[dict]:
        """
        Searches trials based on the diagnosis_search_term.
        The context for this agent should be ONLY the diagnosis string.
        Attachments are ignored for StudienAgent.
        """
        if not diagnosis_search_term or not diagnosis_search_term.strip():
            logger.warning("StudienAgent received empty diagnosis search term. Skipping search.")
            return []
        
        logger.info(f"StudienAgent responding to diagnosis term: '{diagnosis_search_term}'")
        raw_studies_data = self._fetch_studies_from_api(diagnosis_search_term)
        
        enriched_studies = []
        for study_data_item in raw_studies_data:
            processed = self._process_and_enrich_study(study_data_item)
            if processed:
                enriched_studies.append(processed)
        
        if not enriched_studies:
            logger.info("No studies found or processed after API call.")
            return []
            
        sorted_studies = self._sort_studies_by_overall_distance(enriched_studies)
        logger.info(f"StudienAgent returning {len(sorted_studies)} studies.")
        return sorted_studies