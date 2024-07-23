from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import folium
from folium import Marker
from folium.plugins import MarkerCluster
import io
import base64

app = Flask(__name__)


model = joblib.load('tsunami_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_tsunami(magnitude, depth, latitude, longitude):
    input_data = [[magnitude, depth, latitude, longitude]]
    scaled_features = scaler.transform(input_data)
    tsunami_prediction = model.predict_proba(scaled_features)[0][1]
    
    if tsunami_prediction > 0.9:
        tsunami_severity = "High"
    elif tsunami_prediction > 0.7:
        tsunami_severity = "Moderate"
    else:
        tsunami_severity = "Low"
        
    return tsunami_prediction, tsunami_severity

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        magnitude = float(request.form['magnitude'])
        depth = float(request.form['depth'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        
        
        tsunami_probability, tsunami_severity = predict_tsunami(magnitude, depth, latitude, longitude)
        
        
        return redirect(url_for('result', magnitude=magnitude, depth=depth, latitude=latitude, longitude=longitude, 
                                tsunami_probability=tsunami_probability, tsunami_severity=tsunami_severity))
    
    return render_template('index.html')

@app.route('/result')
def result():
    magnitude = float(request.args.get('magnitude'))
    depth = float(request.args.get('depth'))
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))
    tsunami_probability = float(request.args.get('tsunami_probability'))
    tsunami_severity = request.args.get('tsunami_severity')

    
    map_center = [latitude, longitude]
    m = folium.Map(location=map_center, zoom_start=10)

    
    severity_colors = {
        "High": "red",
        "Moderate": "orange",
        "Low": "green"
    }

   
    folium.Marker(
        location=map_center,
        popup=f"Location: {latitude}, {longitude}<br>Severity: {tsunami_severity}<br>Probability: {tsunami_probability:.2f}",
        icon=folium.Icon(color=severity_colors[tsunami_severity])
    ).add_to(m)

    
    map_html = m._repr_html_()
    
    return render_template('result.html', map_html=map_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
