import osmnx as ox
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib import transforms
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd
import sqlite3
import warnings
import requests
import xml.etree.ElementTree as ET
import sys
import re
import os
import time
import json
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")
os.makedirs('static/images', exist_ok=True)
os.makedirs('cache', exist_ok=True)

ox.settings.use_cache = True
ox.settings.cache_folder = './cache'
ox.settings.log_console = False

DB_PATH = 'history.db'
update_time_str = "正在獲取中..."
MACAU_TZ = ZoneInfo('Asia/Macau')

# ==========================================
# 字體與站點設置
# ==========================================
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC', 
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC', 
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
    'Arial Unicode MS'
]
plt.rcParams['axes.unicode_minus'] = False

STATION_COORDS = {
    "紀念孫中山公園": {"lat": 22.216088, "lon": 113.544967}, "黑沙環": {"lat": 22.207104, "lon": 113.560261},
    "大炮臺": {"lat": 22.196739, "lon": 113.542226}, "外港碼頭": {"lat": 22.196783, "lon": 113.558233},
    "媽閣": {"lat": 22.185970, "lon": 113.530564}, "大潭山": {"lat": 22.158936, "lon": 113.568968},
    "東亞運大馬路": {"lat": 22.155145, "lon": 113.542488}, "澳門大學": {"lat": 22.131157, "lon": 113.546909},
    "九澳": {"lat": 22.134651, "lon": 113.580201}, "路環": {"lat": 22.117241, "lon": 113.551101},
    "友誼大橋北": {"lat": 22.190346, "lon": 113.563335}, "友誼大橋南": {"lat": 22.180513, "lon": 113.563346},
    "澳門大橋北": {"lat": 22.188002, "lon": 113.573184}, "澳門大橋南": {"lat": 22.17338, "lon": 113.57310},
    "嘉樂庇總督大橋": {"lat": 22.179965, "lon": 113.545396}, "西灣大橋": {"lat": 22.175653, "lon": 113.534625},
    "蓮花大橋": {"lat": 22.139996, "lon": 113.549312}
}

BRIDGE_STATIONS = ["友誼大橋北", "友誼大橋南", "澳門大橋北", "澳門大橋南", "嘉樂庇總督大橋", "西灣大橋", "蓮花大橋"]

# ==========================================
# 資料庫管理
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS weather_history
                 (time_str TEXT, timestamp REAL, station TEXT, 
                  temp REAL, humidity REAL, wind_speed REAL, rain REAL)''')
    conn.commit()
    conn.close()

def save_to_db(stations, current_time_str):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    today_str = datetime.now(MACAU_TZ).strftime('%Y-%m-%d')
    c.execute("DELETE FROM weather_history WHERE time_str NOT LIKE ?", (today_str + '%',))
    
    now_ts = time.time()
    for name, data in stations.items():
        if any(data.get(k) is not None for k in ['temp_now', 'humidity', 'wind_speed', 'rain_h']):
            c.execute('''INSERT INTO weather_history 
                         (time_str, timestamp, station, temp, humidity, wind_speed, rain) 
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (current_time_str, now_ts, name, data.get('temp_now'), 
                       data.get('humidity'), data.get('wind_speed'), data.get('rain_h')))
    conn.commit()
    conn.close()

# ==========================================
# 折線圖與柱狀圖批量生成
# ==========================================
def generate_all_line_charts():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM weather_history ORDER BY timestamp", conn)
    conn.close()

    if df.empty: return
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    for station_name in STATION_COORDS.keys():
        st_df = df[df['station'] == station_name]
        if st_df.empty: continue

        is_bridge = station_name in BRIDGE_STATIONS
        metrics = ['wind_speed'] if is_bridge else ['temp', 'humidity', 'wind_speed', 'rain']

        for col in metrics:
            valid_df = st_df.dropna(subset=[col])
            
            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor('white')
            
            if valid_df.empty:
                ax.text(0.5, 0.5, '暫無數據 No Data', transform=ax.transAxes, 
                        fontsize=24, color='gray', ha='center', va='center')
                ax.set_title(f"{station_name} - 今日變化", fontsize=16, pad=15)
                ax.set_axis_off()
                ax.text(0.99, 0.98, f"更新時間 Update Time: {update_time_str}", transform=ax.transAxes, fontsize=14, va='top', ha='right', color='#333333')
                plt.tight_layout()
                plt.savefig(f"static/images/line_{station_name}_{col}.png", dpi=120, bbox_inches='tight')
                plt.close(fig)
                continue
            
            if col == 'rain':
                ax.bar(valid_df['datetime'], valid_df[col], width=pd.Timedelta(minutes=4), color='blue', alpha=0.8)
            else:
                line_color = 'red' if col == 'temp' else ('skyblue' if col == 'humidity' else 'black')
                ax.plot(valid_df['datetime'], valid_df[col], linewidth=1.5, color=line_color)
            
            ax.relim()
            ax.autoscale_view()
            ymin, ymax = ax.get_ylim()
            y_buffer = (ymax - ymin) * 0.15 if ymax != ymin else 1
            new_ymin = (ymin - y_buffer) if col == 'temp' else max(0, ymin - y_buffer)
            ax.set_ylim(new_ymin, ymax + y_buffer)

            ax.grid(True, linestyle='--', color='gray', alpha=0.6)
            ax.xaxis.set_major_formatter(DateFormatter('%HH'))
            
            ylabel_map = {
                'temp': '溫度 Temperature (°C)', 'humidity': '濕度 Humidity (%)', 
                'wind_speed': '風速 Wind Speed (km/h)', 'rain': '降水量 Rainfall (mm)'
            }
            ax.set_ylabel(ylabel_map.get(col, col), fontsize=14, color='#333333')
            ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=14, color='#333333', labelpad=10)

            ax.text(0.01, 0.98, station_name, transform=ax.transAxes, fontsize=14, va='top', ha='left', color='#333333')
            ax.text(0.99, 0.98, f"更新時間 Update Time: {update_time_str}", transform=ax.transAxes, fontsize=14, va='top', ha='right', color='#333333')

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#333333')
                
            plt.tight_layout()
            plt.savefig(f"static/images/line_{station_name}_{col}.png", dpi=120, bbox_inches='tight')
            plt.close(fig)

# ==========================================
# 下載真實 OSM 底圖與大橋
# ==========================================
print("正在初始化下載地圖底圖與 OSM 大橋...")
bbox = (113.50, 22.05, 113.65, 22.25)
zhuhai, macau, water, bridges = None, None, None, None

try:
    zhuhai = ox.geocode_to_gdf("Zhuhai, China")
    macau = ox.geocode_to_gdf("Macau, China")
    water = ox.features_from_bbox(bbox=bbox, tags={"natural": ["water", "bay", "strait"], "waterway": ["riverbank", "river", "dock"]})
    water = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    water = water[water.geometry.area > 0.000005]
    bridge_tags = {"bridge": "yes", "highway": ["motorway", "trunk", "primary", "secondary"]}
    bridges = ox.features_from_bbox(bbox=bbox, tags=bridge_tags)
    if bridges is not None and not bridges.empty:
        bridges = bridges[bridges.geometry.type.isin(['LineString', 'MultiLineString'])]
        if not macau.empty:
            macau_geom = macau.geometry.unary_union
            bridges = bridges[bridges.intersects(macau_geom)]
except Exception as e:
    print(f"底圖下載失敗: {e}")

if bridges is None: bridges = gpd.GeoDataFrame()

def draw_base_map(ax, bg_color='#C4E1F5'):
    fig = ax.figure
    fig.patch.set_facecolor('white')
    ax.set_facecolor(bg_color)
    if zhuhai is not None: zhuhai.plot(ax=ax, color='#F5ECD7', edgecolor='none', zorder=1)
    if macau is not None: macau.plot(ax=ax, color='#B0B0B0', edgecolor='white', linewidth=1, zorder=2)
    if water is not None: water.plot(ax=ax, color=bg_color, edgecolor='none', zorder=3)
    if bridges is not None and not bridges.empty: bridges.plot(ax=ax, color='#B0B0B0', edgecolor='white', linewidth=1, zorder=3.5)
    ax.set_xlim(113.51, 113.62); ax.set_ylim(22.08, 22.23); ax.set_axis_off()

# ==========================================
# 核心執行函數
# ==========================================
def update_weather_data():
    global update_time_str
    print(f"[{datetime.now(MACAU_TZ).strftime('%H:%M:%S')}] 開始抓取數據並生成圖片...")
    
    try:
        url = "https://xml.smg.gov.mo/c_actualweather.xml"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        res.encoding = 'utf-8'
        root = ET.fromstring(res.content)
        
        pub_date_node = root.find('.//SysPubDate')
        if pub_date_node is not None and pub_date_node.text:
            raw_time = pub_date_node.text.strip()
            try:
                dt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M")
                minute = (dt.minute // 5) * 5
                dt = dt.replace(minute=minute)
                update_time_str = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                update_time_str = raw_time
        else:
            match = re.search(r'<SysPubDate>(.*?)</SysPubDate>', res.text, re.DOTALL)
            if match:
                raw_time = match.group(1).strip()
                try:
                    dt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M")
                    minute = (dt.minute // 5) * 5
                    dt = dt.replace(minute=minute)
                    update_time_str = dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    update_time_str = raw_time
            else:
                update_time_str = datetime.now(MACAU_TZ).strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        update_time_str = datetime.now(MACAU_TZ).strftime('%Y-%m-%d %H:%M')

    stations = {name: coords.copy() for name, coords in STATION_COORDS.items()}
    def clean_name(n): return n.replace("市政", "").replace("氣象站", "").replace("站", "").strip()
    def get_val(el):
        if el is None: return None
        for t in ['dValue', 'Value']:
            v = el.find(f'.//{t}')
            if v is not None and v.text: 
                try: return float(v.text)
                except: pass
        return None

    for st in root.findall('.//WeatherReport/station'):
        n_node = st.find('stationname')
        if n_node is None: continue
        x_name = clean_name(n_node.text.strip().replace('纪','紀').replace('孙','孫').replace('园','園').replace('台','臺').replace('妈','媽').replace('阁','閣').replace('东','東').replace('运','運').replace('学','學').replace('莲','蓮').replace('桥','橋'))
        m_key = next((k for k in STATION_COORDS if clean_name(k) in x_name or x_name in clean_name(k)), None)
        if not m_key: continue
        
        stations[m_key]['temp_max'] = get_val(st.find('Temperature_daily_max'))
        stations[m_key]['temp_now'] = get_val(st.find('Temperature'))
        stations[m_key]['temp_min'] = get_val(st.find('Temperature_daily_min'))
        stations[m_key]['humidity'] = get_val(st.find('Humidity'))
        
        # 🌟 核心修改：精準解析 SMG 的降雨分類 (Type 4 為小時，Type 5 為日累計)
        h_rain, d_rain = 0.0, None
        for el in st.findall('Rainfall'):
            type_node = el.find('Type')
            if type_node is not None and type_node.text:
                r_type = type_node.text.strip()
                val = get_val(el)
                if r_type == '5': # 日累計
                    d_rain = val
                elif r_type == '4': # 小時級
                    h_rain = val
                # Type 3 (分鐘級) 直接忽略
        
        if d_rain is None: d_rain = 0.0
        
        stations[m_key]['rain_h'] = h_rain
        stations[m_key]['rain_d'] = d_rain
        
        for el in st.iter():
            tag = el.tag.split('}')[-1].lower()
            if tag in ['windspeed', 'wind']:
                val = get_val(el)
                if val is not None: stations[m_key]['wind_speed'] = val
            elif tag in ['windgust', 'gust']:
                val = get_val(el)
                if val is not None: stations[m_key]['gust_speed'] = val
            if 'direction' in tag or tag == 'dir':
                if el.text and el.text.strip(): stations[m_key]['wind_dir'] = el.text.strip().upper()
                else:
                    val_node = el.find('.//Value')
                    if val_node is not None and val_node.text: stations[m_key]['wind_dir'] = val_node.text.strip().upper()

    save_to_db(stations, update_time_str)

    def get_temp_c(t):
        if t is None: return "#E0E0E0", "black"
        if t >= 36: return "#FF8A80", "black"
        elif t >= 32.5: return "#FFB4A2", "black"
        elif t >= 28: return "#FFCC80", "black"
        elif t >= 23: return "#FFE082", "black"
        elif t >= 18: return "#FFF59D", "black"
        elif t >= 12.5: return "#C5E1A5", "black"
        elif t >= 8: return "#A7FFEB", "black"
        elif t >= 5: return "#90CAF9", "black"
        else: return "#CE93D8", "black"
    def get_humid_c(h):
        if h is None: return "#E0E0E0", "black"
        if h <= 30: return "#FFCC80", "black"
        elif h <= 50: return "#FFF59D", "black"
        elif h <= 85: return "#A5D6A7", "black"
        elif h <= 95: return "#90CAF9", "black"
        else: return "#CE93D8", "black"
    def get_hourly_rain_c(a):
        if a is None or a == 0: return "#EEEEEE", "black"
        elif a < 20: return "#B3E5FC", "black"
        elif a < 50: return "#FFF59D", "black"
        elif a < 70: return "#FFCC80", "black"
        elif a < 100: return "#FF8A80", "black"
        else: return "#CE93D8", "black"
    def get_daily_rain_c(a):
        if a is None or a == 0: return "#EEEEEE", "black"
        elif a < 25: return "#B3E5FC", "black"
        elif a < 50: return "#FFF59D", "black"
        elif a < 100: return "#FFCC80", "black"
        elif a < 250: return "#FF8A80", "black"
        else: return "#CE93D8", "black"
    def get_wind_c(s):
        if s is None: return "#E0E0E0", "black"
        if s < 20: return "#FFFFFF", "black"
        elif s <= 40: return "#A5D6A7", "black"
        elif s <= 62: return "#FFF59D", "black"
        elif s <= 117: return "#FFCC80", "black"
        else: return "#FF8A80", "black"

    t_legend = [("極端酷熱", "#FF8A80"), ("酷熱", "#FFB4A2"), ("炎熱", "#FFCC80"), ("溫暖", "#FFE082"), ("和暖", "#FFF59D"), ("清涼", "#C5E1A5"), ("寒冷", "#A7FFEB"), ("嚴寒", "#90CAF9"), ("極端嚴寒", "#CE93D8")]
    h_legend = [("極為乾燥", "#FFCC80"), ("乾燥", "#FFF59D"), ("舒適", "#A5D6A7"), ("潮濕", "#90CAF9"), ("極為潮濕", "#CE93D8")]
    rh_legend = [("0-20mm", "#B3E5FC"), ("20-50mm", "#FFF59D"), ("50-70mm", "#FFCC80"), ("70-100mm", "#FF8A80"), (">100mm", "#CE93D8")]
    rd_legend = [("0-25mm", "#B3E5FC"), ("25-50mm", "#FFF59D"), ("50-100mm", "#FFCC80"), ("100-250mm", "#FF8A80"), (">250mm", "#CE93D8")]
    w_legend = [("0-3級", "#FFFFFF"), ("4-5級", "#A5D6A7"), ("6-7級", "#FFF59D"), ("8-11級", "#FFCC80"), ("12級", "#FF8A80")]

    tasks = [
        ('temp_max', '最高氣溫', get_temp_c, t_legend), ('temp_now', '現時氣溫', get_temp_c, t_legend), ('temp_min', '最低氣溫', get_temp_c, t_legend),
        ('humidity', '相對濕度', get_humid_c, h_legend), ('rain_h', '一小時降水量', get_hourly_rain_c, rh_legend),
        ('rain_d', '今日累計降水', get_daily_rain_c, rd_legend), ('wind_speed', '風速', None, w_legend),
        ('gust_speed', '陣風', None, w_legend), ('wind_dir', '風向', None, None)
    ]

    for key, title, c_func, legend_data in tasks:
        fig, ax = plt.subplots(figsize=(10, 12))
        draw_base_map(ax)
        for name in STATION_COORDS.keys():
            if key in ['temp_max', 'temp_now', 'temp_min', 'humidity', 'rain_h', 'rain_d'] and name in BRIDGE_STATIONS:
                continue 
            data = stations[name]
            val = data.get(key)
            if val is None: continue 
            lon, lat = data['lon'], data['lat']
            if key in ['wind_speed', 'gust_speed']:
                b_c, t_c = get_wind_c(val)
                val_str = f"{val:g}"
                ax.annotate(val_str, xy=(lon, lat), ha='center', va='center', fontsize=11, color='black', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=b_c, edgecolor="#FFFFFF", linewidth=1.5), zorder=4)
            elif key == 'wind_dir':
                dir_map = {'N':0, 'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE':135, 'SSE':157.5, 'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}
                try: deg = float(val)
                except ValueError: deg = dir_map.get(str(val).upper(), 0)
                
                angle = deg % 360 
                
                coords = np.array([[0, 1.5], [-0.8, -0.8], [0, -0.3], [0.8, -0.8]])
                t = transforms.Affine2D().rotate_deg(angle).scale(0.002).translate(lon, lat) + ax.transData
                ax.add_patch(Polygon(coords, closed=True, facecolor='#2C3E50', edgecolor='white', linewidth=0.8, transform=t, zorder=5))
                ax.plot(lon, lat, marker='o', color='white', markersize=2.5, markeredgecolor='#2C3E50', markeredgewidth=0.8, zorder=6)
            else:
                b_c, t_c = c_func(val)
                val_str = f"{val:g}"
                ax.annotate(val_str, xy=(lon, lat), ha='center', va='center', fontsize=11, color='black', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=b_c, edgecolor="white", linewidth=1.5), zorder=4)
        if legend_data:
            patches = [mpatches.Patch(color=c, label=l, edgecolor='#A0A0A0') for l, c in legend_data]
            ax.legend(handles=patches, title="圖例", loc='upper right', title_fontsize=14, fontsize=11, frameon=True, edgecolor='#A0A0A0', facecolor='white', framealpha=0.95)
        ax.text(0.03, 0.03, f"更新時間：{update_time_str}", transform=ax.transAxes, fontsize=12, color='#333333', va='bottom', ha='left', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='#A0A0A0', alpha=0.9), zorder=5)
        plt.title(f"澳門氣象中心 - {title}分佈圖", fontsize=18, pad=20, bbox=dict(facecolor='white', edgecolor='none', pad=12, boxstyle='round,pad=0.5'))
        plt.tight_layout()
        img_filename = f"static/images/{key}.png"
        plt.savefig(img_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

    generate_all_line_charts()
    with open('status.json', 'w', encoding='utf-8') as f:
        json.dump({"update_time": update_time_str}, f, ensure_ascii=False)
    print(f"✅ 全部圖片生成完成！時間：{update_time_str}")

if __name__ == '__main__':
    init_db()
    update_weather_data()
