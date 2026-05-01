# Flight Search Script using Amadeus API
# Requirements: requests, amadeus (install via pip)
# You need to sign up for a free Amadeus developer account to get your API key/secret:
# https://developers.amadeus.com/self-service-apis

import requests
import datetime

# Replace with your Amadeus API credentials
def get_amadeus_access_token(client_id, client_secret):
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def search_flights(access_token, origin, destination, date):
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date,
        "adults": 1,
        "nonStop": False,
        "max": 10
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def main():
    # Toronto (YYZ) to Seattle (SEA) on May 1st
    client_id = "YOUR_AMADEUS_CLIENT_ID"
    client_secret = "YOUR_AMADEUS_CLIENT_SECRET"
    origin = "YYZ"
    destination = "SEA"
    date = "2026-05-01"

    print("Getting Amadeus access token...")
    access_token = get_amadeus_access_token(client_id, client_secret)
    print("Searching for flights...")
    results = search_flights(access_token, origin, destination, date)
    offers = results.get("data", [])
    for offer in offers:
        price = offer["price"]["total"]
        itinerary = offer["itineraries"][0]["segments"]
        dep = itinerary[0]["departure"]["at"]
        arr = itinerary[-1]["arrival"]["at"]
        print(f"Price: ${price}, Departure: {dep}, Arrival: {arr}")

if __name__ == "__main__":
    main()
