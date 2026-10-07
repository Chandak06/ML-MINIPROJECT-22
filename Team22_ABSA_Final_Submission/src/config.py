ASPECT_COLUMNS = [
    "service",
    "rooms",
    "location",
    "value",
    "cleanliness",
    "sleep_quality",
    "business_service",
]

TARGET_COLUMNS = ["overall_rating"] + ASPECT_COLUMNS

ASPECT_DISPLAY_NAMES = {
    "overall_rating": "Overall",
    "service": "Service",
    "rooms": "Rooms",
    "location": "Location",
    "value": "Value",
    "cleanliness": "Cleanliness",
    "sleep_quality": "Sleep Quality",
    "business_service": "Business Service",
}
