from helpers.settings_loader import load_internal_settings

# ============================ 1. TRIP ============================
NUM_TRAVELLERS = 1                            # Applies to flights, trains/buses and hotels
START_CITY = "Stuttgart"
HOME_AIRPORTS = ["STR", "FRA", "FKB"]
TRAVEL_RANGE = ["2027-02-01", "2027-02-15"]    # Inclusive window; departure date is optimized
TOTAL_DAYS = 10                               # Exact calendar days; return on the last day

# ========================= 2. DESTINATIONS =======================
DESTINATIONS = ["London", "Paris", "Madrid", "Barcelona", "Rome"]
NUM_DESTINATIONS = 3                          # Cities to visit (excluding start)
MUST_VISIT_DESTINATIONS = []                  # Subset of DESTINATIONS
MIN_DAYS_PER_CITY = 2                         # Minimum days per destination
MIN_NIGHTS_BY_CITY = {}                       # Optional overrides, e.g. {"Rome": 3}
MIN_USABLE_HOURS_PER_CITY = 9.0               # Hard minimum of daytime hours per stay

# ========================= 3. LIMITS =============================
MAX_BUDGET = 800.0                             # EUR
MAX_CO2 = 800.0                               # kg
MAX_TRANSFER_HOURS = 20.0                     # Per connection, including layovers

# ========================= 4. HOTELS =============================
OPTIMIZE_HOTELS = True                        # Include hotel planning
HOTEL_MIN_STARS = 1                           # 1 to 5
HOTEL_MIN_GUEST_RATING = 3.5                  # On a 5.0 scale
HOTEL_MIN_REVIEWS_COUNT = 30
HOTEL_MAX_PRICE_PER_NIGHT = 70.0             # EUR incl. taxes/fees
HOTEL_INCLUDE_UNSTARRED = True               # Allow unclassified when HOTEL_MIN_STARS > 1

# Load internal settings (do not edit below)
load_internal_settings(globals())
