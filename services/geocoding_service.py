# services/geocoding_service.py
import logging
from typing import Union, Tuple
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

import config

logger = logging.getLogger(__name__)

GEOCODE_CACHE = {}
GEOLOCATOR = Nominatim(user_agent=config.GEOCODER_USER_AGENT)

def get_geopoint(location_string: str) -> Union[Tuple[float, float], None]:
    """Geocodes a location string to (latitude, longitude) with caching."""
    if not location_string or not location_string.strip():
        return None
    normalized_location = location_string.strip().lower()

    if normalized_location in GEOCODE_CACHE:
        return GEOCODE_CACHE[normalized_location]

    logger.info(f"Geocoding: '{location_string}'")
    try:
        location_data = GEOLOCATOR.geocode(location_string, timeout=config.GEOCODE_TIMEOUT)
        if location_data:
            geopoint = (location_data.latitude, location_data.longitude)
            GEOCODE_CACHE[normalized_location] = geopoint
            logger.info(f"Geocoded '{location_string}' to {geopoint}")
            return geopoint
        else:
            logger.warning(f"Could not geocode: '{location_string}' - location not found.")
            GEOCODE_CACHE[normalized_location] = None
            return None
    except GeocoderTimedOut:
        logger.error(f"Geocoding timed out for: '{location_string}'")
        GEOCODE_CACHE[normalized_location] = None # Cache failure
        return None
    except GeocoderUnavailable:
        logger.error(f"Geocoding service unavailable for: '{location_string}'")
        GEOCODE_CACHE[normalized_location] = None # Cache failure
        return None
    except Exception as e:
        logger.error(f"Geocoding general error for '{location_string}': {e}", exc_info=True)
        GEOCODE_CACHE[normalized_location] = None # Cache failure
        return None

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float | None:
    """Calculates geodesic distance in kilometers between two geopoints."""
    try:
        return geodesic(point1, point2).km
    except Exception as e:
        logger.debug(f"Could not calculate distance between {point1} and {point2}: {e}")
        return None