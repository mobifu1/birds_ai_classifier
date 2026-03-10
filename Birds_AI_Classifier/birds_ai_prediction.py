import os
import sqlite3
import datetime
import json
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import requests
from flask import Flask, render_template_string, request, url_for
from waitress import serve

# --- KONFIGURATION NEU ---
APP_VERSION = "Version 1.0-R"
FLASK_PORT = 5001
DB_FILE = "birds_stats.db"
WEATHER_CONFIG_FILE = "weather_config.json"

app = Flask(__name__)

# --- STYLE CSS CONSTANT ---
CSS_STYLE = """
<style>
    body { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        text-align: center; 
        padding: 0; 
        margin: 0;
        background-color: #121212; 
        color: #e0e0e0; 
    }
    .container { 
        width: 90%; 
        max-width: 1200px; 
        margin: 0 auto; 
        padding: 20px; 
        background-color: #1e1e1e;
        border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); 
    }
    .footer {
        margin-top: 30px;
        font-size: 0.8em;
        color: #555;
        border-top: 1px solid #333;
        padding-top: 10px;
    }
    .header-container {
        position: sticky;
        top: 0;
        background-color: #1e1e1e;
        z-index: 1000;
        padding: 10px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 20px;
    }
    h1 { color: #ffffff; margin: 0; font-size: 1.8em; }
    h2 { color: #4fc3f7; margin-top: 30px; border-bottom: 1px solid #333; padding-bottom: 10px;}
    
    .dashboard-controls {
        background: #263238;
        border: 1px solid #37474f;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
    }
    .dashboard-controls label {
        font-size: 1.1em;
        font-weight: bold;
    }
    .dashboard-controls select {
        padding: 8px;
        background: #1e1e1e;
        color: white;
        border: 1px solid #555;
        border-radius: 4px;
        font-size: 1em;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .metric-card {
        background: #263238;
        border: 1px solid #37474f;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
    }
    .metric-card h3 {
        margin-top: 0;
        color: #ffb74d;
        border-bottom: 1px dashed #555;
        padding-bottom: 10px;
    }
    .chart-container {
        margin-top: 20px;
        background: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .chart-container img {
        max-width: 100%;
        height: auto;
        border-radius: 4px;
    }
    
    table { width: 100%; margin: 20px auto; border-collapse: collapse; }
    th { background-color: #0d47a1; color: white; padding: 10px; text-align: left; }
    td { padding: 10px; border-bottom: 1px solid #333; }
    tr:hover { background-color: #2c2c2c; }
</style>
"""

# --- HILFSFUNKTIONEN ---
def load_weather_config():
    if os.path.exists(WEATHER_CONFIG_FILE):
        try:
            with open(WEATHER_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Fehler beim Laden von weather_config.json: {e}")
    return {}

def get_db_data(days=7):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d 00:00:00')
    try:
        query = f"""
            SELECT 
                CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
                timestamp,
                confidence
            FROM detections 
            WHERE timestamp >= '{cutoff_date}'
        """
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['datetime'].dt.hour
            df['date'] = df['datetime'].dt.date
    except Exception as e:
        print(f"Datenbankfehler: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# --- VORHERSAGE FUNKTIONEN ---

def predict_rush_hour(df):
    """Vorhersage der Frequenz (Rush-Hour) der Vögel basierend auf Stunden."""
    # Gruppiere nach Stunde und berechne Durchschnittliche Besuche pro Stunde
    if df.empty:
        return None, ""
    
    unique_days = df['date'].nunique()
    if unique_days == 0: unique_days = 1
    
    hourly_counts = df.groupby('hour').size() / unique_days
    
    # Erstelle ein Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e1e1e')
    ax.bar(hourly_counts.index, hourly_counts.values, color='#4fc3f7')
    ax.set_title('Durchschnittliche Besuche pro Stunde (Rush-Hour)', color='white')
    ax.set_xlabel('Uhrzeit', color='white')
    ax.set_ylabel('Ø Besuche', color='white')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.tick_params(colors='white')
    ax.set_facecolor('#1e1e1e')
    
    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', facecolor='#1e1e1e')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    
    busiest_hour = hourly_counts.idxmax() if not hourly_counts.empty else "N/A"
    return busiest_hour, chart_url

def predict_species_probability(df, target_hour=None):
    """Berechnet die Wahrscheinlichkeit, welche Art in einer bestimmten Stunde auftaucht."""
    if df.empty:
        return {}, ""
    
    if target_hour is None:
        target_hour = datetime.datetime.now().hour
        
    hour_df = df[df['hour'] == target_hour]
    if hour_df.empty:
        return {}, ""
        
    species_counts = hour_df['species'].value_counts()
    total = species_counts.sum()
    probabilities = {sp: (count / total * 100) for sp, count in species_counts.items()}
    
    # Optional: Kuchendiagramm für die Top 5
    top5 = species_counts.head(5)
    if top5.sum() < total:
        top5['Andere'] = total - top5.sum()
        
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')
    ax.pie(top5.values, labels=top5.index, autopct='%1.1f%%', startangle=90, textprops={'color':"w"})
    ax.set_title(f'Wahrscheinlichkeit um {target_hour:02d}:00 Uhr', color='white')
    
    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', facecolor='#1e1e1e')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    
    return probabilities, chart_url

def analyze_disturbance(df):
    """Identifiziert Störereignisse (z.B. durch Elstern/Katzen) und die Zeit bis zum nächsten Besuch."""
    # Placeholder: Wir suchen nach Tieren, die als evtl. störend gelten (müssen definiert werden)
    disturbers = ['Elster', 'Eichelhäher', 'Katze', 'Sperber']
    if df.empty: return "Keine Daten."
    
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    events = []
    
    for i in range(len(df_sorted)):
        row = df_sorted.iloc[i]
        if row['species'] in disturbers:
            # Suche nächsten Besuch eines nicht-Störers
            for j in range(i+1, len(df_sorted)):
                next_row = df_sorted.iloc[j]
                if next_row['species'] not in disturbers:
                    gap = (next_row['datetime'] - row['datetime']).total_seconds() / 60.0
                    events.append({'disturber': row['species'], 'gap_mins': gap})
                    break
                    
    if not events:
        return "Bisher keine Störereignisse nachgewiesen."
        
    avg_gap = sum([e['gap_mins'] for e in events]) / len(events)
    return f"Im Schnitt wird das Futterhaus nach einem Störereignis für {avg_gap:.1f} Minuten gemieden."


def fetch_weather_data(config):
    """Holt aktuelle Wetterdaten von der Weatherunderground API."""
    if not config or "API-Key" not in config or "Station-ID" not in config:
        return None, "Wetter-Konfiguration unvollständig."
        
    api_key = config["API-Key"]
    station_id = config["Station-ID"]
    
    # URL für Weather Underground PWS Current Conditions
    url = f"https://api.weather.com/v2/pws/observations/current?stationId={station_id}&format=json&units=m&apiKey={api_key}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            obs = data.get("observations", [{}])[0]
            metric = obs.get("metric", {})
            
            temp = metric.get("temp", "N/A")
            precip = metric.get("precipRate", 0.0)
            
            # Simple heuristische Vorhersage basierend auf dem aktuellen Wetter
            impact = "Normaler erwarteter Andrang."
            if isinstance(temp, (int, float)) and temp < 5.0:
                impact = f"Es ist kalt ({temp}°C). Rechne mit 30% mehr Futterhaus-Besuchen."
            elif isinstance(precip, (int, float)) and precip > 0:
                impact = f"Es regnet ({precip} mm/h). Vögel suchen verstärkt Schutz und Nahrung."
            elif isinstance(temp, (int, float)) and temp > 25.0:
                impact = f"Es ist warm ({temp}°C). Aktivität verlagert sich vermutlich in die frühen Morgenstunden."
                
            return data, impact
        else:
            return None, f"API Fehler {response.status_code}: Bitte überprüfe den API-Key für Station {station_id}."
    except Exception as e:
        return None, f"Verbindungsfehler zur Wetter-API: {e}"


# --- ROUTES ---
@app.route('/')
def dashboard():
    # Holt das Zeitfenster aus dem Request, Standard = 7 Tage
    try:
        days = int(request.args.get('days', 7))
    except ValueError:
        days = 7
    if days < 1: days = 1
    if days > 30: days = 30
    
    df = get_db_data(days)
    weather_config = load_weather_config()
    
    # 1. Rush Hour
    busiest_hour, rush_hour_chart = predict_rush_hour(df)
    
    # 2. Species Probability für die nächste Stunde
    next_hour = (datetime.datetime.now().hour + 1) % 24
    probs, prob_chart = predict_species_probability(df, target_hour=next_hour)
    
    # 3. Weather Integration
    weather_text = "Keine Wetterstation konfiguriert."
    weather_data_html = ""
    if weather_config:
        w_data, w_impact = fetch_weather_data(weather_config)
        
        weather_text = f"<strong>Vorhersage:</strong> {w_impact}"
        if w_data:
            metric = w_data.get("observations", [{}])[0].get("metric", {})
            t = metric.get("temp", "N/A")
            p = metric.get("precipRate", "0.0")
            h = w_data.get("observations", [{}])[0].get("humidity", "N/A")
            weather_data_html = f"<div style='margin-bottom: 10px; font-size: 0.9em; color: #aaa;'>Aktuell: {t}°C | Niederschlag: {p} mm/h | Luftfeuchtigkeit: {h}%</div>"
        else:
            # Fallback wenn API Key nicht geht
            weather_data_html = f"<div style='margin-bottom: 10px; font-size: 0.9em; color: #ff9800;'>Hinweis: {w_impact} <br>(Für echte Daten gültigen API-Key in weather_config.json eintragen)</div>"
            # Mock Vorhersage für die Demo
            weather_text = "<strong>Muster (Demo):</strong> An regnerischen oder sehr kalten Tagen weicht das Futterverhalten stark von sonnigen Tagen ab. Vögel fressen dann oft mehr und in konzentrierteren Abständen."
    
    # 4. Disturbance Analysis
    disturbance_text = analyze_disturbance(df)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vogel-Verhaltens-Vorhersage (AI)</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
        <meta http-equiv="refresh" content="300">
        {{ css_style|safe }}
    </head>
    <body>
        <div class="container">
            <div class="header-container">
                <h1>🔮 Vogel-Verhaltens-Vorhersage (AI)</h1>
            </div>
            
            <form method="GET" action="/" class="dashboard-controls">
                <label for="days">Analyse-Zeitraum:</label>
                <select name="days" id="days" onchange="this.form.submit()">
                    <option value="1" {% if days == 1 %}selected{% endif %}>Letzte 24 Stunden</option>
                    <option value="3" {% if days == 3 %}selected{% endif %}>Letzte 3 Tage</option>
                    <option value="7" {% if days == 7 %}selected{% endif %}>Letzte 7 Tage</option>
                    <option value="14" {% if days == 14 %}selected{% endif %}>Letzte 14 Tage</option>
                    <option value="30" {% if days == 30 %}selected{% endif %}>Letzte 30 Tage</option>
                </select>
            </form>
            
            <div class="metrics-grid">
                <!-- 1. Rush Hour -->
                <div class="metric-card">
                    <h3>1. Rush-Hour Vorhersage</h3>
                    <p>Wann ist der größte Andrang am Futterhaus zu erwarten?</p>
                    {% if rush_hour_chart %}
                        <div style="font-size: 1.2em; color: #81d4fa; margin-bottom: 15px;">
                            Beste Zeit für Nachschub: <strong>Vor {{ busiest_hour }}:00 Uhr</strong>
                        </div>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{{ rush_hour_chart }}" alt="Rush Hour Chart">
                        </div>
                    {% else %}
                        <p>Nicht genug Daten für eine Vorhersage.</p>
                    {% endif %}
                </div>
                
                <!-- 2. Species Probability -->
                <div class="metric-card">
                    <h3>2. Arten-Wahrscheinlichkeit</h3>
                    <p>Wer kommt voraussichtlich bis um <strong>{{ next_hour }}:00 Uhr</strong> vorbei?</p>
                    {% if prob_chart %}
                        <ul>
                        {% for sp, prob in probs.items() %}
                            {% if prob > 5 %}
                            <li>{{ sp }}: <strong>{{ "%.1f"|format(prob) }}%</strong></li>
                            {% endif %}
                        {% endfor %}
                        </ul>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{{ prob_chart }}" alt="Probability Chart">
                        </div>
                    {% else %}
                        <p>Zu dieser Uhrzeit gab es noch keine Sichtungen in den letzten {{ days }} Tagen.</p>
                    {% endif %}
                </div>
                
                <!-- 3. Weather Integration -->
                <div class="metric-card">
                    <h3>3. Wetter-Verknüpfung</h3>
                    <p>Korrelation von Futterverhalten und Wetter ({{ weather_config.get('Service-Name', 'Service') }} API).</p>
                    <div style="background: #1e1e1e; padding: 10px; border-radius: 5px; border-left: 4px solid #4fc3f7;">
                        {{ weather_data_html|safe }}
                        {{ weather_text|safe }}
                    </div>
                </div>
                
                <!-- 4. Disturbance Analysis -->
                <div class="metric-card">
                    <h3>4. Störereignisse</h3>
                    <p>Verhaltensänderung nach Auftauchen von z.B. Elstern, Raubtieren oder Katzen.</p>
                    <div style="background: #1e1e1e; padding: 10px; border-radius: 5px; border-left: 4px solid #ff5252;">
                        {{ disturbance_text|safe }}
                    </div>
                </div>
            </div>
            
            <div class="footer">
                {{ version }}
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html, 
                                  css_style=CSS_STYLE, 
                                  days=days, 
                                  busiest_hour=busiest_hour,
                                  rush_hour_chart=rush_hour_chart,
                                  next_hour=next_hour,
                                  probs=probs,
                                  prob_chart=prob_chart,
                                  weather_config=weather_config,
                                  weather_data_html=weather_data_html,
                                  weather_text=weather_text,
                                  disturbance_text=disturbance_text,
                                  version=APP_VERSION)

if __name__ == '__main__':
    print(f"Starte AI-Prediction Webserver auf Port {FLASK_PORT}...")
    serve(app, host='0.0.0.0', port=FLASK_PORT)
