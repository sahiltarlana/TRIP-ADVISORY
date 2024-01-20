import pandas as pd
import requests
import re
import random

cnt_col = 0
cnt_mod = 0
cnt_hot = 0
cnt_tot = 0
destinations_data = ["Mysore",
    "Hampi",
    "Coorg ",
    "Gokarna",
    "Udupi",
    "Chikmagalur",
    "Badami",
    "Hassan",
    "Shivamogga (Shimoga)",
    "Mangalore",
    "Belur",
    "Halebidu",
    "Bijapur",
    "Aihole",
    "Pattadakal",
    "Murudeshwar",
    "Jog Falls",
    "Karwar",
    "Dandeli",
    "Chitradurga",
    "Srirangapatna",
    "Madikeri",
    "Bandipur National Park",
    "Nagarhole National Park",
    "Ramanagara",
    "Agumbe",
    "Sakleshpur",
    "Tumkur",
    "Bidar",
    "Ooty",
     "Kodaikanal",
    "Goa",
    "Madurai",
    "Kochi",
    "Pondicherry",
    "Kanyakumari",
    "Rameswaram",
    "Thekkady",
    "Mahabalipuram",
    "Wayanad" ,
    "Badami",
    "Varkala",
    "Kumarakom",
    "Nandi Hills",
    "Mararikulam",
    "Hogenakkal Falls",
    "Araku Valley",
    "Yercaud",
    "Coonoor",
    "Yelagiri",
    "Gudalur",
    "Kovalam",
    "Kollam",
    "Malpe",
    "Dhanushkodi",
    "Coimbatore",
    "Kalpetta,",
    "Srirangapatna",
    "Vellore",
    "Chettinad",
    "Gavi",
    "Shivanasamudra Falls",
    "Nagercoil",
    "Athirapally Falls",
    "Warangal",
    "Mahabaleshwar",
    "Horsley Hills",
    "Sravanabelagola"
]

def categorize_temperature(temperature_celsius):
    global cnt_col, cnt_mod, cnt_hot, cnt_tot
    try:
        temperature = int(temperature_celsius)
        if temperature < 28:
            cnt_col += 1
            cnt_tot += 1
            return "Cold"
        elif 28 <= temperature <= 30:
            cnt_mod += 1
            cnt_tot += 1
            return "Moderate"
        else:
            cnt_hot += 1
            cnt_tot += 1
            return "Hot"
    except ValueError:
        return "Invalid Temperature"
def get_coordinates(city_name):
    url = "http://www.mapquestapi.com/geocoding/v1/address"
    params = {
        "key": "OmWxCjrAutZJei8jFYjf1we6eEbCuiEu",
        "location": city_name,
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "results" in data and data["results"]:
        first_result = data["results"][0]
        location = first_result["locations"][0]
        lat = location["latLng"]["lat"]
        lng = location["latLng"]["lng"]
        return lat, lng
    else:
        return None



temp = [] 
cord_lat=[]
cord_lon=[]# List to store temperature data
budget=[]
for i in range (0,68):
    budget.append(random.choice(["expensive", "moderate", "low"]))

for city in destinations_data:
    latlon=get_coordinates(city)
    cord_lat.append(latlon[0])
    cord_lon.append(latlon[1])
    # if (latlon != None):
    #     print(f"the cord of {city} are {latlon}")
    # else:
    #     print(city)
    url = f"https://wttr.in/{city.replace(' ', '+')}?format=%t"
    response = requests.get(url)
    temperature_celsius = response.text
    l = temperature_celsius.split()
    text = l[0]
    temperature_match = re.search(r'([-+]?\d+)', text)
    if temperature_match:
        temperature_celsius = int(temperature_match.group(1))
    else:
        print(f"No temperature data found for {city}")
        temperature_celsius = "N/A"  # Use a placeholder value
    weather_category = categorize_temperature(temperature_celsius)
    temp.append(temperature_celsius)
     
print(cnt_col)
print(cnt_mod)
print(cnt_hot)
print(cnt_tot)
df = pd.DataFrame({
"Destination": destinations_data,
"Latitude":cord_lat,
"Longitude":cord_lon,
"weather": [categorize_temperature(temp_val) for temp_val in temp],
"cost":budget
})

df.to_excel("destinations.xlsx", index=False)
print("Excel file 'destinations1.xlsx' has been created.")
