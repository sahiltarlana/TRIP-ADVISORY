import folium, wikipedia
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure 
from flask import Response,Flask,send_file,render_template,request
import io
from PIL import Image
import textwrap
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import math
import base64
import folium, wikipedia
import requests


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index1.html')

@app.route("/form")
def form():
    return render_template('form.html')

@app.route('/recomendations',methods=['POST'])
def getvalue():
    budget = request.form['budget']
    weather = request.form['weather']
    min_dis = request.form['mindistance']
    max_dis = request.form['maxdistance']
    location=request.form['location']
    rec=[]
    rec = getrec(weather,min_dis,max_dis,budget)
    return render_template('rec.html',cities=rec)


@app.route("/show",methods=['POST'])
def show():
    cities_chosen=request.form.getlist("city")
    cities_chosen.append("NITK")
    cities=cities_chosen
    output=generate_tsp_graph(cities_chosen)
    image_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
    return render_template('pass.html', image_data=image_base64)

@app.route("/map",methods=['POST'])
def map():
    cities_chosen_m=request.form.getlist("city")
    cities_chosen_m.append("NITK")
    create_map(cities_chosen_m)
    return render_template('my_map.html')


#@app.route('/plot')



g = []
cities=[]
dist=0

class Destination:
    def __init__(self, name, weather, distance, cost):
        self.name = name
        self.weather = weather
        self.distance = distance
        self.cost = cost

# loading data to excel sheet
def get_destination_data_from_excel():
    df = pd.read_excel("destinations1.xlsx")
    # print(df)
    destinations = []
    your_lat = 13.0108
    your_lon = 74.7943
    for index, row in df.iterrows():
        name = row["Destination"]
        weather = row["weather"]
        cost = row["cost"]
        coordinates = (row["Latitude"], row["Longitude"])
        distance = calculate_distance1(coordinates, (your_lat, your_lon))
        "user can replace it with thier own latitude and longitude"
        destinations.append(Destination(name, weather, distance, cost))
    return destinations
        
def get_coordinates1(city_name):
    df = pd.read_excel("destinations1.xlsx")
    coordinates = None
    for index, row in df.iterrows():
        if row["Destination"] == city_name:
            coordinates = (row["Latitude"], row["Longitude"])
        if city_name == "NITK":
            coordinates = (13.0108, 74.7943)
    return coordinates

def calculate_distance1(coords1, coords2):
    return geodesic(coords1, coords2).kilometers

def calculate_distance2(city1, city2):
    df = pd.read_excel("destinations1.xlsx")
    for index, row in df.iterrows():
        if city1 == row["Destination"]:
            cord1 = (row["Latitude"], row["Longitude"])
        if city2 == row["Destination"]:
            cord2 = (row["Latitude"], row["Longitude"])
        if city1 == "NITK":
            cord1 = (13.0108, 74.7943)
        if city2 == "NITK":
            cord2 = (13.0108, 74.7943)
    distance = calculate_distance1(cord1, cord2)
    return distance

def calculate_distance(city1, city2):
    coordinates1 = get_coordinates1(city1)
    coordinates2 = get_coordinates1(city2)

    if coordinates1 is None:
        print(f"Could not find coordinates for {city1}")
        return
    if coordinates2 is None:
        print(f"Could not find coordinates for {city2}")
        return

    lat1, lng1 = coordinates1
    lat2, lng2 = coordinates2
    earth_radius_km = 6371

    lat1_rad = lat1 * (3.141592653589793 / 180)
    lng1_rad = lng1 * (3.141592653589793 / 180)
    lat2_rad = lat2 * (3.141592653589793 / 180)
    lng2_rad = lng2 * (3.141592653589793 / 180)

    dlon = lng2_rad - lng1_rad
    dlat = lat2_rad - lat1_rad
    a = pow(dlat / 2, 2) + math.cos(lat1_rad) * math.cos(lat2_rad) * pow(dlon / 2, 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius_km * c

    return distance

def set_intersection(set1, set2):
    """
    Custom intersection function for sets.
    """
    intersection = set()
    for item in set1:
        if item in set2:
            intersection.add(item)
    return intersection

def filter_destinations1(destinations, attr, value1,value2):
        min_distance=value1 
        max_distance = value2
        return {
            dest
            for dest in destinations
            if min_distance <= dest.distance <= max_distance
        }
        
           
    
def filter_destinations(destinations, attr, value):
        return {dest for dest in destinations if getattr(dest, attr) == value}



indian_destinations = get_destination_data_from_excel()
def getrec(a,b1,b2,c):
        cities=[]
        if a == '':
            weather_based_destinations = set(indian_destinations)
        else:   
            weather_based_destinations = filter_destinations(indian_destinations, "weather", a)
        # print (weather_based_destinations)
        if c == '':
            cost_based_destinations = set(indian_destinations)
        else:   
            cost_based_destinations = filter_destinations(indian_destinations, "cost", c)   
        
            
        intersection_destinations1 = set_intersection(
            weather_based_destinations, cost_based_destinations
        )
        # print(intersection_destinations1)
        if (b1=='' and b2=='') :
            distance_based_destinations = set(indian_destinations)
            
        else:
            
            distance_based_destinations = filter_destinations1(indian_destinations, "distance", int(b1),int(b2))
            
        intersection_destinations2 = set_intersection(
            distance_based_destinations, intersection_destinations1
        )
        # print(intersection_destinations2)

        for dest in intersection_destinations2:
            cities.append(dest.name)
        return cities
    
def get_wikipedia_info(place_name):
    try:
        page = wikipedia.page(place_name)
        return (
            page.summary[:200],
            page.images[0] if page.images else None,
            page.images[1] if len(page.images) >= 2 else None,
        )
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation by choosing the first result
        page = wikipedia.page(e.options[0])
        return page.summary[:200], page.images[0] if page.images else None, page.images[1] if len(page.images) >= 2 else None
    except wikipedia.exceptions.PageError:
         return None, None, None
    

def copy_list(l):
    l1 = []
    for i in l:
        l1.append(i)
    return l1

def pathf(x, num):
    list_path = []
    l = [x]
    def t(l):
        i = l[-1]
        for j in range(num):
            if g[i][j] is not None and g[i][j] != 0:
                if j not in l:
                    l.append(j)
                    if len(l) == num:
                        if g[j][x] is not None:
                            l.append(x)
                            list_path.append(copy_list(l))
                            l.pop()
                        l.pop()
                    else:
                        t(l)
        if l != []:
            l.pop()

    t(l)
    return list_path

def path_length(path):
    len_path = len(path)
    total_distance = 0
    for i in range(len_path - 1):
        total_distance += g[path[i]][path[i + 1]]
    return total_distance

def shortest_path(list_path):
    smallest_sum = -1
    s_path = []
    for i in list_path:
        total_distance = path_length(i)
        if smallest_sum == -1 or smallest_sum > total_distance:
            smallest_sum = total_distance
            s_path = [i]
        elif smallest_sum == total_distance:
            s_path.append(i)
    return s_path

def find_tsp(cities):
    global g ,dist # Access the global variable g
    g = []  # Reinitialize g for each call
    n_cities = len(cities)

    for i in range(n_cities):
        l = []
        for j in range(n_cities):
            l.append(calculate_distance2(cities[i], cities[j]))
        g.append(copy_list(l))

    num = n_cities
    x = num
    list_path = pathf(x - 1, num)
    p_shortest = shortest_path(list_path)
    shortest_dist = path_length(p_shortest[0])
    dist=shortest_dist
    tsp = []
    path = p_shortest[0]
    len_p = len(path)

    for j in range(len_p - 1):
        tsp.append(cities[path[j]])

    tsp.append(cities[path[-1]])
    return tsp
    
    
def generate_tsp_graph(cities):
    result_string = ""
    tsp=[]
    tsp =find_tsp(cities)
    n_cities=len(tsp)-1
    for j in range(n_cities):
        result_string = result_string + (tsp[j] + " -->  ")
    result_string = result_string + (tsp[len(tsp)-1])
    max_width=85
    result_string = textwrap.fill(result_string, width=max_width)
        
    G_tsp = nx.Graph()
    n_cities=len(tsp)-1
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            G_tsp.add_edge(cities[i], cities[j], weight=g[i][j])
    start_node="NITK"

    # Find the TSP path using the built-in TSP solver from networkx
    tsp_path_g = nx.approximation.traveling_salesman_problem(G_tsp, cycle=True)
    total_length_g = sum(G_tsp[u][v]['weight'] for u, v in zip(tsp_path_g[:-1], tsp_path_g[1:]))


# Print the TSP path and its length
    print("TSP Path with existing in build algorithm :", tsp_path_g)
    print("Total Length with existing in build algorithm :", total_length_g)
    print("TSP Path with Our algorithm :", tsp)
    print("TSP Path length with Our algorithm :", dist)
    
    
    cit=copy_list(tsp)
    cit.append(cit[0])
    tsp_path=cit
    path_edges = [(tsp_path[i], tsp_path[i + 1]) for i in range(len(tsp_path) - 1)]
    tsp_graph = G_tsp.subgraph(path_edges)
    edge_styles = []
    for edge in G_tsp.edges():
        if edge in path_edges:
            edge_styles.append('solid')
        else:
            edge_styles.append('dotted')
    plt.figure(figsize=(20, 15))
    fig,ax = plt.subplots()
    pos_tsp = nx.spring_layout(G_tsp, seed=42)
    labels_tsp = {node: node for node in G_tsp.nodes()}

    tsp_edges = [(tsp_path[i], tsp_path[i + 1]) for i in range(len(tsp_path) - 1)] + [
    (tsp_path[-1], tsp_path[0])
    ]
    edge_labels = {(u, v): f"{G_tsp[u][v]['weight']:.2f} km" for u, v in G_tsp.edges()}
    edge_styles = [
    (u, v, {"style": "dotted", "color": "yellow", "width": 1.0, "alpha": 0.5})
    for u, v in G_tsp.edges()
    ]
    tsp_edge_styles = [
    (u, v, {"style": "solid", "color": "red", "width": 3.0}) for u, v in tsp_edges
    ]

    nx.draw_networkx_nodes(G_tsp, pos_tsp, node_size=500, node_color="skyblue", ax=ax)
    nx.draw_networkx_labels(
    G_tsp, pos_tsp, labels=labels_tsp, font_size=10, font_color="black", ax=ax
    )

    nx.draw_networkx_edges(G_tsp, pos_tsp, edgelist=tsp_edge_styles)
    nx.draw_networkx_edges(G_tsp, pos_tsp, edgelist=edge_styles, style="dotted", edge_color="black", width=1.0, alpha=0.5)


    nx.draw_networkx_edges(G_tsp, pos_tsp, edgelist=tsp_edges, edge_color="red", width=4)

    ax.text(0.5, 0, result_string, transform=ax.transAxes, ha="center")
    plt.axis("off")
    plt.title("Optimal Path for your Trip")
    output = io.BytesIO()
    plt.savefig(output, format='png')
    output.seek(0)
    return output


def create_map(cities) :
    tsp=[]
    tsp=find_tsp(cities)
    coordinates = []
    sum1 = [0, 0]
    for city in tsp:
        c = get_coordinates1(city)
        coordinates.append(c)
        sum1[0] += c[0]
        sum1[1] += c[1]
    n_cities=len(tsp)
    map_center = ((sum1[0] / (n_cities + 1)), sum1[1] / (n_cities + 1))
    my_map = folium.Map(location=map_center, zoom_start=5)

    # Add a PolyLine to represent the path between the cities
    route = []

    for i in range(n_cities):
        print(tsp[i])
        if tsp[i] == "NITK":
            summary, image_url1, image_url2 = get_wikipedia_info(
                "National Institute of Technology, Karnataka"
            )
        else:
            summary, image_url1, image_url2 = get_wikipedia_info(tsp[i] + ", India")

        # Get weather data for the city
        weather_data = get_weather_data(tsp[i])

        popup_html = f"<h4>{tsp[i]}</h4><p>{summary}</p>"
        # Add weather information to the popup
        if weather_data:
            popup_html += f"<p><b>Weather</b>: {weather_data}</p>"

        if image_url1:
            popup_html += f'<img src="{image_url1}" alt="{tsp[i]}" style="width:100%;height:auto;">'
        if image_url2:
            popup_html += f'<img src="{image_url2}" alt="{tsp[i]}" style="width:100%;height:auto;">'

        folium.Marker(
            location=coordinates[i], popup=folium.Popup(popup_html, max_width=300)
        ).add_to(my_map)
        route.append(coordinates[i])

    folium.PolyLine(route, color="blue", weight=2.5, opacity=1, dash_array="10, 5").add_to(my_map)

    # Add a LayerControl to toggle the display of the route
    folium.LayerControl().add_to(my_map)

    my_map.save(r"templates\my_map.html")



def get_wikipedia_info(place_name):
    try:
        page = wikipedia.page(place_name)
        return (
            page.summary[:200],
            page.images[0] if page.images else None,
            page.images[1] if len(page.images) >= 2 else None,
        )
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation by choosing the first result
        page = wikipedia.page(e.options[0])
        return (
            page.summary[:200],
            page.images[0] if page.images else None,
            page.images[1] if len(page.images) >= 2 else None,
        )
    except wikipedia.exceptions.PageError:
        return None, None, None

def get_weather_data(city_name):
    try:
        # URL to fetch weather data in JSON format
        url = f"https://wttr.in/{city_name.replace(' ', '+')}?format=%C+%t+%w+%P"

        response = requests.get(url)
        data = response.text

        return data
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None





# Calculate the total length of the TSP path







if __name__ == '__main__':
    app.run(debug=True)


