import requests
import datetime

def search_location(name: str, count: int = 1, language: str = "en", apikey: str = None):
    """
    Calls the Open-Meteo Geocoding API to search for a location.

    :param name: The location name or postal code to search for (minimum 2 characters required).
    :param count: The number of search results to return (default: 10, max: 100).
    :param language: Language code for localization (default: "en").
    :param apikey: Optional API key for commercial use.
    :return: List of matching locations with name, latitude, and longitude or an error message.
    """
    if len(name) < 2:
        raise ValueError("Search term must be at least 2 characters long.")
    
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": name,
        "count": count,
        "format": "json",
        "language": language
    }
    
    if apikey:
        params["apikey"] = apikey
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        return {"error": True, "reason": f"HTTP {response.status_code}: {response.text}"}
    
    data = response.json()
    
    if "results" in data:
        return [{"name": loc["name"], "latitude": loc["latitude"], "longitude": loc["longitude"]} for loc in data["results"]]
    
    return {"error": True, "reason": "No results found."}


def get_weather(latitude: float, longitude: float, date: str):
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "timezone": "auto",
        "forecast_days": 7
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return {"error": "Failed to fetch data", "status_code": response.status_code}
    
    data = response.json()
    if "hourly" not in data or "temperature_2m" not in data["hourly"]:
        return {"error": "Invalid response format"}
    
    # Extract hourly time and temperature
    time_list = data["hourly"]["time"]
    temp_list = data["hourly"]["temperature_2m"]
    
    # Convert date to datetime object
    input_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    # Find the temperatures for the requested date
    result = []
    for i, timestamp in enumerate(time_list):
        if timestamp.startswith(str(input_date)):
            result.append({"time": timestamp, "temperature": temp_list[i]})
    
    if not result:
        return {"error": "No data available for the given date"}
    
    return result

# Example usage
if __name__ == "__main__":
    search_results = search_location("Berlin")
    print(search_results)

    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))
    date = input("Enter date (YYYY-MM-DD): ")
    
    weather_data = get_weather(latitude, longitude, date)
    print(weather_data)
    
    # Example call
    example_result = get_weather(52.52, 13.41, "2023-02-04")
    print("Example result:", example_result)



