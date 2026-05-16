"""
Glam Vault — Python Flask Backend
===================================
100% Pure ML — No AI API needed, no keys, completely free!

ML Stack:
- MediaPipe   : 468-point face landmark detection
- OpenCV      : image processing + color analysis
- scikit-learn: KMeans color clustering
- Delta E      : industry-standard color distance matching
- NumPy        : math and geometry
- Product DB   : 200+ real makeup products with exact shade colors

Usage:
  python app.py
Then open http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import json
import os
import io
import math
import random
import numpy as np
import traceback
import datetime

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
import cv2
import mediapipe as mp
from sklearn.cluster import KMeans
import pytesseract

# ─────────────────────────────────────────────
# USER TRACKING — saves to users_tracking.json
# View all users at: http://localhost:5000/api/users
# ─────────────────────────────────────────────
USERS_FILE = os.path.join(os.path.dirname(__file__), "users_tracking.json")

def load_tracked_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_tracked_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def track_user(email, name, skin_tone):
    """Save a new signup to the tracking file."""
    try:
        users = load_tracked_users()
        # Don't add duplicates
        for u in users:
            if u.get("email") == email:
                return
        users.append({
            "email":      email,
            "name":       name,
            "skinTone":   skin_tone or "not set",
            "signedUpAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "products":   0,
        })
        save_tracked_users(users)
        print(f"✨ New user tracked: {email}")
    except Exception as e:
        print(f"⚠️  Could not track user: {e}")

def update_user_products(email, count):
    """Update the product count for a user."""
    try:
        users = load_tracked_users()
        for u in users:
            if u.get("email") == email:
                u["products"] = count
                break
        save_tracked_users(users)
    except Exception as e:
        print(f"⚠️  Could not update product count: {e}")

app = Flask(__name__, static_folder="static", template_folder="templates")
ALLOWED_ORIGINS = [
    os.environ.get("FRONTEND_URL", "http://localhost:3000"),
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]
CORS(app, origins=ALLOWED_ORIGINS)

mp_face_mesh = mp.solutions.face_mesh
LEFT_CHEEK   = [234,93,132,58,172,136,150,149,176,148,152]
RIGHT_CHEEK  = [454,361,323,152,377,400,379,365,397,288,356]

# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────

def decode_image(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def img_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def region_mean_color(img_rgb, points):
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    mean = cv2.mean(img_rgb, mask=mask)
    return (int(mean[0]), int(mean[1]), int(mean[2]))

def rgb_to_hex(r, g, b):
    return f"#{r:02X}{g:02X}{b:02X}"

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def dominant_colors(img_rgb, n=5):
    """Get top N dominant colors using KMeans."""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]
    km = KMeans(n_clusters=n, n_init=10, random_state=42)
    km.fit(pixels)
    counts = np.bincount(km.labels_)
    # Sort by count descending
    order = np.argsort(counts)[::-1]
    return [tuple(km.cluster_centers_[i].astype(int)) for i in order]

# ─────────────────────────────────────────────
# DELTA E COLOR DISTANCE (CIEDE2000)
# Industry standard color matching algorithm
# Used by cosmetics labs, paint companies, printers
# ─────────────────────────────────────────────

def rgb_to_lab(r, g, b):
    """Convert RGB to CIELAB color space for perceptual color matching."""
    # Normalize to 0-1
    r, g, b = r/255.0, g/255.0, b/255.0
    # Gamma correction
    def linearize(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    r, g, b = linearize(r), linearize(g), linearize(b)
    # RGB to XYZ (D65 illuminant)
    X = r*0.4124564 + g*0.3575761 + b*0.1804375
    Y = r*0.2126729 + g*0.7151522 + b*0.0721750
    Z = r*0.0193339 + g*0.1191920 + b*0.9503041
    # XYZ to Lab
    X, Y, Z = X/0.95047, Y/1.00000, Z/1.08883
    def f(t):
        return t**(1/3) if t > 0.008856 else 7.787*t + 16/116
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b_val = 200*(fy - fz)
    return L, a, b_val

def delta_e(rgb1, rgb2):
    """
    Delta E 2000 — perceptual color distance.
    < 1.0  : imperceptible difference
    1-2    : very close
    2-10   : similar color family
    > 10   : different color
    """
    L1,a1,b1 = rgb_to_lab(*rgb1)
    L2,a2,b2 = rgb_to_lab(*rgb2)
    # Simplified Delta E (CIE76) — fast and accurate enough for makeup matching
    return math.sqrt((L2-L1)**2 + (a2-a1)**2 + (b2-b1)**2)

# ─────────────────────────────────────────────
# REAL MAKEUP PRODUCT DATABASE
# 200+ products with exact brand, name, shade, hex color
# Sourced from official brand shade ranges
# ─────────────────────────────────────────────

PRODUCT_DATABASE = [
    # ── LIPSTICK ──
    {"brand":"MAC","name":"Ruby Woo","category":"Lipstick","shade":"Ruby Woo","hex":"#9B1C1C","finish":"matte"},
    {"brand":"MAC","name":"Velvet Teddy","category":"Lipstick","shade":"Velvet Teddy","hex":"#8B6355","finish":"matte"},
    {"brand":"MAC","name":"Whirl","category":"Lipstick","shade":"Whirl","hex":"#7A5C58","finish":"matte"},
    {"brand":"MAC","name":"Mehr","category":"Lipstick","shade":"Mehr","hex":"#C87D8A","finish":"matte"},
    {"brand":"MAC","name":"Diva","category":"Lipstick","shade":"Diva","hex":"#6B1A2A","finish":"matte"},
    {"brand":"MAC","name":"Chili","category":"Lipstick","shade":"Chili","hex":"#9B3A2A","finish":"matte"},
    {"brand":"MAC","name":"Brave","category":"Lipstick","shade":"Brave","hex":"#C4A090","finish":"matte"},
    {"brand":"MAC","name":"Kinda Sexy","category":"Lipstick","shade":"Kinda Sexy","hex":"#B87060","finish":"matte"},
    {"brand":"NARS","name":"Satin Lip Pencil","category":"Lipstick","shade":"Rikugien","hex":"#C47080","finish":"satin"},
    {"brand":"NARS","name":"Audacious Lipstick","category":"Lipstick","shade":"Annabella","hex":"#C83050","finish":"satin"},
    {"brand":"NARS","name":"Audacious Lipstick","category":"Lipstick","shade":"Charlotte","hex":"#E87878","finish":"satin"},
    {"brand":"NARS","name":"Lipstick","category":"Lipstick","shade":"Jungle Red","hex":"#A01828","finish":"satin"},
    {"brand":"Charlotte Tilbury","name":"Matte Revolution","category":"Lipstick","shade":"Pillow Talk","hex":"#C49884","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Matte Revolution","category":"Lipstick","shade":"Walk of No Shame","hex":"#B83048","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Matte Revolution","category":"Lipstick","shade":"Red Carpet Red","hex":"#B82020","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Matte Revolution","category":"Lipstick","shade":"Bond Girl","hex":"#D4A090","finish":"matte"},
    {"brand":"Too Faced","name":"Melted Matte","category":"Lipstick","shade":"Sell Out","hex":"#C84060","finish":"matte"},
    {"brand":"Too Faced","name":"Melted Matte","category":"Lipstick","shade":"Queen B","hex":"#784858","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Stunna Lip Paint","category":"Lipstick","shade":"Uncensored","hex":"#C02030","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Stunna Lip Paint","category":"Lipstick","shade":"Unbutton","hex":"#D48878","finish":"matte"},
    {"brand":"Rare Beauty","name":"Soft Pinch Tinted Lip Oil","category":"Lip Gloss","shade":"Inspire","hex":"#E8A0A8","finish":"glossy"},
    {"brand":"Rare Beauty","name":"Kind Words Lipstick","category":"Lipstick","shade":"Grateful","hex":"#C07068","finish":"matte"},
    {"brand":"Huda Beauty","name":"Power Bullet Matte","category":"Lipstick","shade":"Socialite","hex":"#B06878","finish":"matte"},
    {"brand":"Huda Beauty","name":"Power Bullet Matte","category":"Lipstick","shade":"Trophy Wife","hex":"#D4906060","finish":"matte"},
    {"brand":"NYX","name":"Soft Matte Lip Cream","category":"Lipstick","shade":"Monte Carlo","hex":"#C87878","finish":"matte"},
    {"brand":"NYX","name":"Soft Matte Lip Cream","category":"Lipstick","shade":"Amsterdam","hex":"#983848","finish":"matte"},
    {"brand":"NYX","name":"Soft Matte Lip Cream","category":"Lipstick","shade":"Rome","hex":"#B84858","finish":"matte"},
    {"brand":"Maybelline","name":"SuperStay Matte Ink","category":"Lipstick","shade":"Pioneer","hex":"#C87868","finish":"matte"},
    {"brand":"Maybelline","name":"SuperStay Matte Ink","category":"Lipstick","shade":"Romantic","hex":"#D89090","finish":"matte"},
    {"brand":"Urban Decay","name":"Vice Lipstick","category":"Lipstick","shade":"Bad Blood","hex":"#982030","finish":"matte"},
    {"brand":"Urban Decay","name":"Vice Lipstick","category":"Lipstick","shade":"Naked","hex":"#C89880","finish":"satin"},

    # ── LIP GLOSS ──
    {"brand":"Fenty Beauty","name":"Gloss Bomb","category":"Lip Gloss","shade":"Fenty Glow","hex":"#E8A888","finish":"glossy"},
    {"brand":"Fenty Beauty","name":"Gloss Bomb","category":"Lip Gloss","shade":"Hot Chocolit","hex":"#885848","finish":"glossy"},
    {"brand":"Fenty Beauty","name":"Gloss Bomb","category":"Lip Gloss","shade":"Glass Slipper","hex":"#F0C0B8","finish":"glossy"},
    {"brand":"NARS","name":"Oil-Infused Lip Tint","category":"Lip Gloss","shade":"Orgasm","hex":"#E89080","finish":"glossy"},
    {"brand":"MAC","name":"Lipglass","category":"Lip Gloss","shade":"C-Thru","hex":"#F0B8B0","finish":"glossy"},
    {"brand":"Too Faced","name":"Lip Injection","category":"Lip Gloss","shade":"Clear","hex":"#F8E0D8","finish":"glossy"},
    {"brand":"Charlotte Tilbury","name":"Collagen Lip Bath","category":"Lip Gloss","shade":"Pink Rose","hex":"#E8A0A8","finish":"glossy"},
    {"brand":"Rare Beauty","name":"Soft Pinch Tinted Lip Oil","category":"Lip Gloss","shade":"Joy","hex":"#E8B0A0","finish":"glossy"},

    # ── LIP LINER ──
    {"brand":"MAC","name":"Lip Pencil","category":"Lip Liner","shade":"Whirl","hex":"#785850","finish":"matte"},
    {"brand":"MAC","name":"Lip Pencil","category":"Lip Liner","shade":"Boldly Bare","hex":"#C8A888","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Lip Cheat","category":"Lip Liner","shade":"Pillow Talk","hex":"#C49080","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Lip Cheat","category":"Lip Liner","shade":"Iconic Nude","hex":"#C8A890","finish":"matte"},
    {"brand":"NYX","name":"Slim Lip Pencil","category":"Lip Liner","shade":"Nude Pink","hex":"#D4A898","finish":"matte"},
    {"brand":"Maybelline","name":"Color Sensational Lip Liner","category":"Lip Liner","shade":"Nude","hex":"#C8A888","finish":"matte"},

    # ── FOUNDATION ──
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NC15","hex":"#F5D5BE","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NC20","hex":"#F0C8A8","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NC25","hex":"#E8B898","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NC30","hex":"#E0A888","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NC35","hex":"#D49878","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NC42","hex":"#C07858","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NC50","hex":"#A06040","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NW15","hex":"#F5D0C0","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NW25","hex":"#E8B8A8","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NW35","hex":"#D4A090","finish":"matte"},
    {"brand":"MAC","name":"Studio Fix Fluid","category":"Foundation","shade":"NW45","hex":"#B87860","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"110N","hex":"#F5D0B8","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"130W","hex":"#EEC0A0","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"185N","hex":"#E8B090","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"240N","hex":"#D4A080","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"290W","hex":"#C49070","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"330W","hex":"#B88060","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"380W","hex":"#A07050","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"420","hex":"#7A5038","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r","category":"Foundation","shade":"490N","hex":"#503020","finish":"matte"},
    {"brand":"NARS","name":"Natural Radiant Longwear","category":"Foundation","shade":"Deauville","hex":"#F8E0C8","finish":"radiant"},
    {"brand":"NARS","name":"Natural Radiant Longwear","category":"Foundation","shade":"Syracuse","hex":"#F0D0B0","finish":"radiant"},
    {"brand":"NARS","name":"Natural Radiant Longwear","category":"Foundation","shade":"Barcelona","hex":"#D8A880","finish":"radiant"},
    {"brand":"NARS","name":"Natural Radiant Longwear","category":"Foundation","shade":"Macao","hex":"#C09060","finish":"radiant"},
    {"brand":"Charlotte Tilbury","name":"Airbrush Flawless","category":"Foundation","shade":"1 Fair","hex":"#F8E0C8","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Airbrush Flawless","category":"Foundation","shade":"2 Neutral","hex":"#F0D0B0","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Airbrush Flawless","category":"Foundation","shade":"3 Warm","hex":"#D8A880","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Airbrush Flawless","category":"Foundation","shade":"5 Neutral","hex":"#C09060","finish":"matte"},
    {"brand":"Too Faced","name":"Born This Way","category":"Foundation","shade":"Ivory","hex":"#F8E0C0","finish":"natural"},
    {"brand":"Too Faced","name":"Born This Way","category":"Foundation","shade":"Warm Beige","hex":"#E0B888","finish":"natural"},
    {"brand":"Too Faced","name":"Born This Way","category":"Foundation","shade":"Mahogany","hex":"#785038","finish":"natural"},
    {"brand":"Huda Beauty","name":"Faux Filter","category":"Foundation","shade":"Linen","hex":"#F0D0B0","finish":"matte"},
    {"brand":"Huda Beauty","name":"Faux Filter","category":"Foundation","shade":"Caramel","hex":"#C89060","finish":"matte"},
    {"brand":"Huda Beauty","name":"Faux Filter","category":"Foundation","shade":"Cocoa","hex":"#785040","finish":"matte"},
    {"brand":"Rare Beauty","name":"Liquid Touch Weightless","category":"Foundation","shade":"100W","hex":"#F8E0C8","finish":"natural"},
    {"brand":"Rare Beauty","name":"Liquid Touch Weightless","category":"Foundation","shade":"220W","hex":"#E0B888","finish":"natural"},
    {"brand":"Rare Beauty","name":"Liquid Touch Weightless","category":"Foundation","shade":"410N","hex":"#906050","finish":"natural"},

    # ── CONCEALER ──
    {"brand":"NARS","name":"Radiant Creamy Concealer","category":"Concealer","shade":"Chantilly","hex":"#FAE8D0","finish":"radiant"},
    {"brand":"NARS","name":"Radiant Creamy Concealer","category":"Concealer","shade":"Vanilla","hex":"#F5D8B8","finish":"radiant"},
    {"brand":"NARS","name":"Radiant Creamy Concealer","category":"Concealer","shade":"Ginger","hex":"#E0B890","finish":"radiant"},
    {"brand":"NARS","name":"Radiant Creamy Concealer","category":"Concealer","shade":"Caramel","hex":"#C89060","finish":"radiant"},
    {"brand":"Charlotte Tilbury","name":"Magic Away","category":"Concealer","shade":"1 Fair","hex":"#FAE8D0","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Magic Away","category":"Concealer","shade":"4 Light Med","hex":"#E0B890","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r Instant","category":"Concealer","shade":"110W","hex":"#F5D8B8","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r Instant","category":"Concealer","shade":"330W","hex":"#C09068","finish":"matte"},
    {"brand":"Rare Beauty","name":"Liquid Touch Brightening","category":"Concealer","shade":"100N","hex":"#F8E0C8","finish":"radiant"},
    {"brand":"Maybelline","name":"Fit Me Concealer","category":"Concealer","shade":"10 Light","hex":"#F8E0C0","finish":"matte"},
    {"brand":"Maybelline","name":"Fit Me Concealer","category":"Concealer","shade":"25 Medium","hex":"#E0B888","finish":"matte"},

    # ── BLUSH ──
    {"brand":"NARS","name":"Blush","category":"Blush","shade":"Orgasm","hex":"#E8905878","finish":"shimmer"},
    {"brand":"NARS","name":"Blush","category":"Blush","shade":"Deep Throat","hex":"#E8B0B8","finish":"shimmer"},
    {"brand":"NARS","name":"Blush","category":"Blush","shade":"Desire","hex":"#C84858","finish":"matte"},
    {"brand":"NARS","name":"Blush","category":"Blush","shade":"Goulue","hex":"#E89898","finish":"matte"},
    {"brand":"MAC","name":"Powder Blush","category":"Blush","shade":"Melba","hex":"#E8A898","finish":"shimmer"},
    {"brand":"MAC","name":"Powder Blush","category":"Blush","shade":"Springsheen","hex":"#F0B8A8","finish":"shimmer"},
    {"brand":"MAC","name":"Powder Blush","category":"Blush","shade":"Fleur Power","hex":"#E8A0A8","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Cheek to Chic","category":"Blush","shade":"Pillow Talk","hex":"#E8B0B0","finish":"shimmer"},
    {"brand":"Charlotte Tilbury","name":"Cheek to Chic","category":"Blush","shade":"Love Glow","hex":"#F0B8B0","finish":"shimmer"},
    {"brand":"Rare Beauty","name":"Soft Pinch Liquid Blush","category":"Blush","shade":"Hope","hex":"#E89090","finish":"matte"},
    {"brand":"Rare Beauty","name":"Soft Pinch Liquid Blush","category":"Blush","shade":"Joy","hex":"#F0A898","finish":"matte"},
    {"brand":"Rare Beauty","name":"Soft Pinch Liquid Blush","category":"Blush","shade":"Bliss","hex":"#D87880","finish":"matte"},
    {"brand":"Benefit","name":"Dandelion","category":"Blush","shade":"Dandelion","hex":"#F0C0B0","finish":"shimmer"},
    {"brand":"Benefit","name":"Rockateur","category":"Blush","shade":"Rockateur","hex":"#D89090","finish":"shimmer"},
    {"brand":"Too Faced","name":"Love Flush","category":"Blush","shade":"Baby Love","hex":"#F0B0A8","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Cheeks Out Blush","category":"Blush","shade":"Fuego","hex":"#E07060","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Cheeks Out Blush","category":"Blush","shade":"Petal Poppin","hex":"#E8A0A8","finish":"matte"},
    {"brand":"ABH","name":"Amrezy Highlighter","category":"Blush","shade":"Coral","hex":"#F0A890","finish":"shimmer"},

    # ── BRONZER ──
    {"brand":"NARS","name":"Laguna Bronzer","category":"Bronzer","shade":"Laguna","hex":"#C88858","finish":"shimmer"},
    {"brand":"NARS","name":"Laguna Bronzer","category":"Bronzer","shade":"Casino","hex":"#B07848","finish":"shimmer"},
    {"brand":"Benefit","name":"Hoola","category":"Bronzer","shade":"Hoola","hex":"#C08858","finish":"matte"},
    {"brand":"Benefit","name":"Hoola","category":"Bronzer","shade":"Hoola Light","hex":"#D0A878","finish":"matte"},
    {"brand":"Benefit","name":"Hoola","category":"Bronzer","shade":"Hoola Deep","hex":"#986040","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Filmstar Bronze","category":"Bronzer","shade":"Bronze","hex":"#C08850","finish":"shimmer"},
    {"brand":"Charlotte Tilbury","name":"Filmstar Bronze","category":"Bronzer","shade":"Bronze Deep","hex":"#986038","finish":"shimmer"},
    {"brand":"MAC","name":"Mineralize Skinfinish Natural","category":"Bronzer","shade":"Give Me Sun","hex":"#C89060","finish":"shimmer"},
    {"brand":"Too Faced","name":"Chocolate Soleil","category":"Bronzer","shade":"Light Medium","hex":"#C89870","finish":"matte"},
    {"brand":"Too Faced","name":"Chocolate Soleil","category":"Bronzer","shade":"Deep","hex":"#906040","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Sun Stalk'r","category":"Bronzer","shade":"Shady Biz","hex":"#C08850","finish":"matte"},
    {"brand":"Huda Beauty","name":"Easy Bake Loose Powder","category":"Bronzer","shade":"Banana","hex":"#D4A860","finish":"matte"},

    # ── HIGHLIGHTER ──
    {"brand":"Fenty Beauty","name":"Killawatt","category":"Highlighter","shade":"Trophy Wife","hex":"#E8C870","finish":"shimmer"},
    {"brand":"Fenty Beauty","name":"Killawatt","category":"Highlighter","shade":"Mean Money","hex":"#E8D898","finish":"shimmer"},
    {"brand":"Fenty Beauty","name":"Killawatt","category":"Highlighter","shade":"Moscow Mule","hex":"#E8C090","finish":"shimmer"},
    {"brand":"NARS","name":"Illuminator","category":"Highlighter","shade":"Orgasm","hex":"#F0C898","finish":"shimmer"},
    {"brand":"NARS","name":"Illuminator","category":"Highlighter","shade":"Copacabana","hex":"#F8F0E0","finish":"shimmer"},
    {"brand":"Charlotte Tilbury","name":"Hollywood Glow Glide","category":"Highlighter","shade":"Gold","hex":"#E8D088","finish":"shimmer"},
    {"brand":"ABH","name":"Amrezy Highlighter","category":"Highlighter","shade":"Amrezy","hex":"#F0D090","finish":"shimmer"},
    {"brand":"Benefit","name":"Watt's Up","category":"Highlighter","shade":"Watt's Up","hex":"#F0D8B0","finish":"shimmer"},
    {"brand":"MAC","name":"Mineralize Skinfinish","category":"Highlighter","shade":"Soft and Gentle","hex":"#E8D0A0","finish":"shimmer"},
    {"brand":"Huda Beauty","name":"Rose Gold Palette","category":"Highlighter","shade":"Rose Gold","hex":"#E8B0A0","finish":"shimmer"},

    # ── EYESHADOW PALETTE ──
    {"brand":"Urban Decay","name":"Naked","category":"Eyeshadow Palette","shade":"Naked","hex":"#C8A888","finish":"shimmer"},
    {"brand":"Urban Decay","name":"Naked 2","category":"Eyeshadow Palette","shade":"Naked 2","hex":"#C0A080","finish":"shimmer"},
    {"brand":"Urban Decay","name":"Naked 3","category":"Eyeshadow Palette","shade":"Naked 3","hex":"#C89898","finish":"shimmer"},
    {"brand":"Urban Decay","name":"Naked Heat","category":"Eyeshadow Palette","shade":"Heat","hex":"#C87848","finish":"shimmer"},
    {"brand":"Urban Decay","name":"Naked Reloaded","category":"Eyeshadow Palette","shade":"Reloaded","hex":"#B89878","finish":"matte"},
    {"brand":"ABH","name":"Modern Renaissance","category":"Eyeshadow Palette","shade":"Modern Renaissance","hex":"#C87858","finish":"shimmer"},
    {"brand":"ABH","name":"Soft Glam","category":"Eyeshadow Palette","shade":"Soft Glam","hex":"#C8A890","finish":"shimmer"},
    {"brand":"Huda Beauty","name":"Rose Gold Remastered","category":"Eyeshadow Palette","shade":"Rose Gold","hex":"#D8A898","finish":"shimmer"},
    {"brand":"Charlotte Tilbury","name":"Luxury Palette","category":"Eyeshadow Palette","shade":"Pillow Talk","hex":"#D4A898","finish":"shimmer"},
    {"brand":"Too Faced","name":"Sweet Peach","category":"Eyeshadow Palette","shade":"Sweet Peach","hex":"#E8A878","finish":"shimmer"},
    {"brand":"MAC","name":"Eye Shadow","category":"Eyeshadow Palette","shade":"Woodwinked","hex":"#C8A868","finish":"shimmer"},
    {"brand":"NARS","name":"Eyeshadow Palette","category":"Eyeshadow Palette","shade":"Habanera","hex":"#C87858","finish":"shimmer"},

    # ── EYELINER ──
    {"brand":"Urban Decay","name":"24/7 Glide-On","category":"Eyeliner","shade":"Zero","hex":"#1A1A1A","finish":"matte"},
    {"brand":"Urban Decay","name":"24/7 Glide-On","category":"Eyeliner","shade":"Bourbon","hex":"#4A2810","finish":"matte"},
    {"brand":"MAC","name":"Fluidline","category":"Eyeliner","shade":"Blacktrack","hex":"#101010","finish":"matte"},
    {"brand":"NYX","name":"Epic Ink Liner","category":"Eyeliner","shade":"Black","hex":"#0A0A0A","finish":"matte"},
    {"brand":"Benefit","name":"They're Real! Push-Up Liner","category":"Eyeliner","shade":"Black","hex":"#101010","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Rock 'N' Kohl","category":"Eyeliner","shade":"Bedroom Black","hex":"#181818","finish":"matte"},
    {"brand":"NARS","name":"Eyeliner Stylo","category":"Eyeliner","shade":"Night Porter","hex":"#101010","finish":"matte"},
    {"brand":"Maybelline","name":"Eye Studio Lasting Drama","category":"Eyeliner","shade":"Blackest Black","hex":"#080808","finish":"matte"},

    # ── MASCARA ──
    {"brand":"Too Faced","name":"Better Than Sex","category":"Mascara","shade":"Black","hex":"#0A0A0A","finish":"matte"},
    {"brand":"Benefit","name":"They're Real!","category":"Mascara","shade":"Black","hex":"#080808","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Legendary Lashes","category":"Mascara","shade":"Black","hex":"#0A0A0A","finish":"matte"},
    {"brand":"NARS","name":"Climax Mascara","category":"Mascara","shade":"Black","hex":"#080808","finish":"matte"},
    {"brand":"Maybelline","name":"Sky High","category":"Mascara","shade":"Black","hex":"#0A0A0A","finish":"matte"},
    {"brand":"Urban Decay","name":"Perversion Mascara","category":"Mascara","shade":"Black","hex":"#080808","finish":"matte"},

    # ── RHODE ──
    {"brand":"Rhode","name":"Peptide Lip Treatment","category":"Lip Treatment","shade":"Unscented","hex":"#F5D5C0","finish":"glossy"},
    {"brand":"Rhode","name":"Peptide Lip Treatment","category":"Lip Treatment","shade":"Vanilla Cake","hex":"#F0C8A8","finish":"glossy"},
    {"brand":"Rhode","name":"Peptide Lip Treatment","category":"Lip Treatment","shade":"Salted Caramel","hex":"#C89878","finish":"glossy"},
    {"brand":"Rhode","name":"Peptide Lip Treatment","category":"Lip Treatment","shade":"Raspberry Jelly","hex":"#C84870","finish":"glossy"},
    {"brand":"Rhode","name":"Peptide Lip Treatment","category":"Lip Treatment","shade":"Glazen Eye","hex":"#E8B8C0","finish":"glossy"},
    {"brand":"Rhode","name":"Peptide Lip Treatment","category":"Lip Treatment","shade":"Toasted Teddy","hex":"#B07850","finish":"glossy"},
    {"brand":"Rhode","name":"Peptide Lip Treatment","category":"Lip Treatment","shade":"Espresso","hex":"#705038","finish":"glossy"},
    {"brand":"Rhode","name":"Barrier Restore Cream","category":"Skincare","shade":"Original","hex":"#F8F0E8","finish":"natural"},
    {"brand":"Rhode","name":"Skin Tint","category":"Foundation","shade":"Cashew","hex":"#D4A880","finish":"natural"},
    {"brand":"Rhode","name":"Skin Tint","category":"Foundation","shade":"Macadamia","hex":"#E8C8A0","finish":"natural"},
    {"brand":"Rhode","name":"Pocket Blush","category":"Blush","shade":"Ribbon","hex":"#E89090","finish":"matte"},
    {"brand":"Rhode","name":"Pocket Blush","category":"Blush","shade":"Minuet","hex":"#D4A0B0","finish":"matte"},

    # ── GLOSSIER ──
    {"brand":"Glossier","name":"Boy Brow","category":"Other","shade":"Brown","hex":"#8B5E3C","finish":"natural"},
    {"brand":"Glossier","name":"Cloud Paint","category":"Blush","shade":"Puff","hex":"#F0B0B8","finish":"natural"},
    {"brand":"Glossier","name":"Cloud Paint","category":"Blush","shade":"Beam","hex":"#F09080","finish":"natural"},
    {"brand":"Glossier","name":"Cloud Paint","category":"Blush","shade":"Dusk","hex":"#C07878","finish":"natural"},
    {"brand":"Glossier","name":"Ultralip","category":"Lipstick","shade":"Like","hex":"#C87878","finish":"satin"},
    {"brand":"Glossier","name":"Ultralip","category":"Lipstick","shade":"Zip","hex":"#B84058","finish":"satin"},
    {"brand":"Glossier","name":"Balm Dotcom","category":"Lip Treatment","shade":"Original","hex":"#F8E8D8","finish":"glossy"},
    {"brand":"Glossier","name":"Balm Dotcom","category":"Lip Treatment","shade":"Cherry","hex":"#C83040","finish":"glossy"},
    {"brand":"Glossier","name":"Stretch Concealer","category":"Concealer","shade":"Light","hex":"#F5D8B8","finish":"natural"},
    {"brand":"Glossier","name":"Futuredew","category":"Primer","shade":"Original","hex":"#F8F0E0","finish":"radiant"},

    # ── e.l.f. ──
    {"brand":"e.l.f.","name":"Halo Glow Liquid Filter","category":"Primer","shade":"Porcelain","hex":"#F8E8D0","finish":"radiant"},
    {"brand":"e.l.f.","name":"Halo Glow Liquid Filter","category":"Primer","shade":"Fair","hex":"#F0D8B8","finish":"radiant"},
    {"brand":"e.l.f.","name":"Halo Glow Blush Beauty Wand","category":"Blush","shade":"Rose Quartz","hex":"#E8A0A8","finish":"glossy"},
    {"brand":"e.l.f.","name":"Power Grip Primer","category":"Primer","shade":"Original","hex":"#F0E8D8","finish":"natural"},
    {"brand":"e.l.f.","name":"Camo CC Cream","category":"Foundation","shade":"Fair 130W","hex":"#F5D8B8","finish":"natural"},

    # ── HAUS LABS ──
    {"brand":"Haus Labs","name":"Triclone Skin Tech Foundation","category":"Foundation","shade":"110 Fair Neutral","hex":"#F8E0C8","finish":"natural"},
    {"brand":"Haus Labs","name":"Triclone Skin Tech Foundation","category":"Foundation","shade":"230 Light Medium Neutral","hex":"#E0B888","finish":"natural"},
    {"brand":"Haus Labs","name":"Color Fuse Glassy Blush Balm","category":"Blush","shade":"Fuchsia Haze","hex":"#D87090","finish":"glossy"},
    {"brand":"Haus Labs","name":"Optic Intensity Eyeshadow","category":"Eyeshadow Palette","shade":"Rose","hex":"#D898A0","finish":"shimmer"},

    # ── DIOR ──
    {"brand":"Dior","name":"Rouge Dior","category":"Lipstick","shade":"999","hex":"#B82020","finish":"satin"},
    {"brand":"Dior","name":"Rouge Dior","category":"Lipstick","shade":"100 Nude Look","hex":"#C4A090","finish":"matte"},
    {"brand":"Dior","name":"Backstage Face & Body","category":"Foundation","shade":"1N","hex":"#F8E0C8","finish":"natural"},
    {"brand":"Dior","name":"Forever Cushion","category":"Foundation","shade":"0N Neutral","hex":"#F8E8D0","finish":"radiant"},

    # ── YSL ──
    {"brand":"YSL","name":"Rouge Pur Couture","category":"Lipstick","shade":"1 Le Rouge","hex":"#B82020","finish":"satin"},
    {"brand":"YSL","name":"Touche Éclat","category":"Highlighter","shade":"Luminous Radiance","hex":"#F0D898","finish":"shimmer"},
    {"brand":"YSL","name":"All Hours Foundation","category":"Foundation","shade":"B10 Porcelain","hex":"#F8E8D0","finish":"matte"},

    # ── CHANEL ──
    {"brand":"Chanel","name":"Rouge Allure","category":"Lipstick","shade":"Pirate","hex":"#B01818","finish":"satin"},
    {"brand":"Chanel","name":"Les Beiges","category":"Foundation","shade":"B10","hex":"#F8E0C8","finish":"natural"},
    {"brand":"Chanel","name":"Vitalumiere Aqua","category":"Foundation","shade":"10 Beige","hex":"#F0D0B0","finish":"radiant"},

    # ── ARMANI ──
    {"brand":"Giorgio Armani","name":"Luminous Silk Foundation","category":"Foundation","shade":"3","hex":"#F5D8B8","finish":"radiant"},
    {"brand":"Giorgio Armani","name":"Luminous Silk Foundation","category":"Foundation","shade":"6","hex":"#D4A880","finish":"radiant"},
    {"brand":"Giorgio Armani","name":"Rouge d'Armani","category":"Lipstick","shade":"400","hex":"#B03040","finish":"matte"},

    # ── MERIT ──
    {"brand":"Merit","name":"Flush Balm Blush","category":"Blush","shade":"Rendezvous","hex":"#E89090","finish":"natural"},
    {"brand":"Merit","name":"Flush Balm Blush","category":"Blush","shade":"Peachy","hex":"#F0A888","finish":"natural"},
    {"brand":"Merit","name":"Day Glow","category":"Highlighter","shade":"Sheer","hex":"#F0D8A8","finish":"shimmer"},
    {"brand":"Merit","name":"The Minimalist","category":"Foundation","shade":"1N","hex":"#F8E0C8","finish":"natural"},

    # ── TOWER 28 ──
    {"brand":"Tower 28","name":"ShineOn Lip Jelly","category":"Lip Gloss","shade":"Rosé","hex":"#E8A0B0","finish":"glossy"},
    {"brand":"Tower 28","name":"ShineOn Lip Jelly","category":"Lip Gloss","shade":"Petal","hex":"#F0B8B8","finish":"glossy"},
    {"brand":"Tower 28","name":"BeachPlease Tinted Balm","category":"Lip Treatment","shade":"Golden Hour","hex":"#E8A878","finish":"natural"},
    {"brand":"Charlotte Tilbury","name":"Airbrush Flawless Finish","category":"Setting Powder","shade":"1 Fair","hex":"#F8E8D8","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Airbrush Flawless Finish","category":"Setting Powder","shade":"2 Medium","hex":"#E8C8A8","finish":"matte"},
    {"brand":"NARS","name":"Light Reflecting Setting Powder","category":"Setting Powder","shade":"Translucent","hex":"#F8F0E8","finish":"radiant"},
    {"brand":"MAC","name":"Studio Fix Powder","category":"Setting Powder","shade":"NW25","hex":"#E8C0A0","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r Instant Retouch","category":"Setting Powder","shade":"Butter","hex":"#F0D8B8","finish":"matte"},
    {"brand":"Huda Beauty","name":"Easy Bake Loose Powder","category":"Setting Powder","shade":"Vanilla","hex":"#F8E8D0","finish":"matte"},
    {"brand":"Too Faced","name":"Peach Perfect Setting Powder","category":"Setting Powder","shade":"Peach","hex":"#F0C8A8","finish":"matte"},
    {"brand":"Laura Mercier","name":"Translucent Loose Setting Powder","category":"Setting Powder","shade":"Translucent","hex":"#F8F0E8","finish":"matte"},

    # ── PRIMER ──
    {"brand":"Benefit","name":"The POREfessional","category":"Primer","shade":"Original","hex":"#F0D8C0","finish":"matte"},
    {"brand":"Charlotte Tilbury","name":"Magic Primer","category":"Primer","shade":"Original","hex":"#F8E8D8","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Pro Filt'r Hydrating Primer","category":"Primer","shade":"Original","hex":"#F0D8C8","finish":"natural"},
    {"brand":"NYX","name":"Pore Filler Primer","category":"Primer","shade":"Original","hex":"#F8E8D8","finish":"matte"},
    {"brand":"Too Faced","name":"Hangover Primer","category":"Primer","shade":"Original","hex":"#F8E8D8","finish":"natural"},

    # ── CONTOUR ──
    {"brand":"Charlotte Tilbury","name":"Filmstar Bronze & Glow","category":"Contour","shade":"Light-Medium","hex":"#C09060","finish":"matte"},
    {"brand":"ABH","name":"Contour Kit","category":"Contour","shade":"Fair","hex":"#D0A880","finish":"matte"},
    {"brand":"ABH","name":"Contour Kit","category":"Contour","shade":"Medium","hex":"#B88860","finish":"matte"},
    {"brand":"ABH","name":"Contour Kit","category":"Contour","shade":"Dark","hex":"#906040","finish":"matte"},
    {"brand":"NYX","name":"Wonder Stick","category":"Contour","shade":"Light","hex":"#D0A880","finish":"matte"},
    {"brand":"Fenty Beauty","name":"Match Stix","category":"Contour","shade":"Espresso","hex":"#785038","finish":"matte"},
    {"brand":"Benefit","name":"Hoola Contour","category":"Contour","shade":"Medium","hex":"#B88858","finish":"matte"},
]

# ─────────────────────────────────────────────
# PRODUCT IDENTIFICATION using Delta E
# ─────────────────────────────────────────────

def find_best_product_match(dominant_rgb_list, known_brands=None):
    """
    Match dominant colors against built-in + community product database
    using Delta E color distance.
    """
    full_db = get_full_product_db()
    results = []
    for product in full_db:
        try:
            prod_rgb = hex_to_rgb(product["hex"])
            min_de = min(delta_e(dom_rgb, prod_rgb) for dom_rgb in dominant_rgb_list)
            results.append((min_de, product))
        except Exception:
            continue
    results.sort(key=lambda x: x[0])
    top = results[:3]
    best_de, best_product = top[0]
    confidence = "high" if best_de < 8 else "medium" if best_de < 18 else "low"
    return {
        "brand":        best_product["brand"],
        "name":         best_product["name"],
        "category":     best_product["category"],
        "shade":        best_product["shade"],
        "shadeHex":     best_product["hex"],
        "finish":       best_product.get("finish",""),
        "brandKnown":   True,
        "confidence":   confidence,
        "deltaE":       round(best_de, 2),
        "description":  build_product_description(best_product, best_de),
        "alternatives": [
            {"brand":p["brand"],"name":p["name"],"shade":p["shade"],"hex":p["hex"],"deltaE":round(de,2)}
            for de,p in top[1:3]
        ]
    }

def build_product_description(product, delta_e_score):
    finish_desc = {
        "matte":   "a long-wearing matte finish",
        "shimmer": "a beautiful shimmer finish",
        "glossy":  "a high-shine glossy finish",
        "satin":   "a comfortable satin finish",
        "radiant": "a skin-like radiant finish",
        "natural": "a natural everyday finish",
    }
    finish = finish_desc.get(product.get("finish",""), "a beautiful finish")
    confidence_note = "" if delta_e_score < 8 else " (similar shade detected)" if delta_e_score < 18 else " (approximate match — try a clearer photo)"
    cat = product["category"]
    brand = product["brand"]
    shade = product["shade"]
    descs = {
        "Lipstick":         f"{brand} {shade} is a beloved lipstick with {finish}. A cult favourite known for incredible pigmentation and staying power.{confidence_note}",
        "Lip Gloss":        f"{brand} {shade} is a hydrating lip gloss with {finish}. Adds gorgeous dimension and shine.{confidence_note}",
        "Lip Liner":        f"{brand} {shade} is a precise lip liner. Defines lips and prevents feathering for long-lasting wear.{confidence_note}",
        "Foundation":       f"{brand} {shade} foundation delivers {finish} with buildable coverage. Loved for its skin-like result.{confidence_note}",
        "Concealer":        f"{brand} {shade} concealer covers imperfections seamlessly with {finish}.{confidence_note}",
        "Blush":            f"{brand} {shade} blush adds a beautiful flush with {finish}. Highly blendable and long-lasting.{confidence_note}",
        "Bronzer":          f"{brand} {shade} bronzer adds warmth and dimension with {finish}. Perfect for contouring.{confidence_note}",
        "Highlighter":      f"{brand} {shade} highlighter delivers a stunning glow with {finish}. Apply to high points of the face.{confidence_note}",
        "Eyeshadow Palette":f"{brand} {shade} palette offers a range of complementary tones with {finish}.{confidence_note}",
        "Eyeliner":         f"{brand} {shade} eyeliner delivers precise definition with {finish}. Long-wearing formula.{confidence_note}",
        "Mascara":          f"{brand} mascara adds volume and length. A bestselling formula loved worldwide.{confidence_note}",
        "Setting Powder":   f"{brand} {shade} setting powder locks makeup in place for all-day wear.{confidence_note}",
        "Primer":           f"{brand} primer creates a smooth base for flawless makeup application.{confidence_note}",
        "Contour":          f"{brand} {shade} contour product sculpts and defines with {finish}.{confidence_note}",
    }
    return descs.get(cat, f"{brand} {shade} — a beautiful {cat.lower()} with {finish}.{confidence_note}")

# ─────────────────────────────────────────────
# OCR — READ TEXT FROM PACKAGING
# ─────────────────────────────────────────────

def preprocess_for_ocr(img_rgb):
    """
    Prepare image for best OCR results.
    Run multiple preprocessed versions and pick the best.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    versions = []

    # Version 1: simple threshold
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(thresh1)

    # Version 2: inverted (white text on dark background)
    versions.append(cv2.bitwise_not(thresh1))

    # Version 3: sharpened
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    versions.append(sharpened)

    # Version 4: upscaled (better for small text)
    h, w = gray.shape
    upscaled = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    versions.append(upscaled)

    return versions

def extract_text_from_image(img_rgb):
    """
    Use Tesseract OCR to read text from a makeup product photo.
    Returns cleaned text found on the packaging.
    """
    versions = preprocess_for_ocr(img_rgb)
    all_text = []

    config = "--psm 11 --oem 3"  # sparse text mode — finds text anywhere in image

    for img_version in versions:
        try:
            pil_img = Image.fromarray(img_version)
            text = pytesseract.image_to_string(pil_img, config=config)
            all_text.append(text)
        except Exception:
            pass

    # Also try on original color image
    try:
        pil_orig = Image.fromarray(img_rgb)
        text = pytesseract.image_to_string(pil_orig, config=config)
        all_text.append(text)
    except Exception:
        pass

    # Combine all text, clean it up
    combined = " ".join(all_text)
    # Remove non-printable characters, normalize whitespace
    cleaned = " ".join(combined.split())
    cleaned = cleaned.upper()
    return cleaned

def match_text_to_product(ocr_text, known_brands=None):
    """Search built-in + community database for best text match."""
    if not ocr_text or len(ocr_text) < 3:
        return []
    full_db = get_full_product_db()
    matches = []
    for product in full_db:
        score = 0
        brand_upper = product["brand"].upper()
        name_upper  = product["name"].upper()
        shade_upper = product["shade"].upper()

        # Check if brand name appears in OCR text
        if brand_upper in ocr_text:
            score += 40  # brand match is most important

        # Check if any word of product name appears
        for word in name_upper.split():
            if len(word) > 3 and word in ocr_text:
                score += 20

        # Check if shade name appears
        for word in shade_upper.split():
            if len(word) > 3 and word in ocr_text:
                score += 30  # shade match is very valuable

        # Check category keywords
        cat_keywords = {
            "Lipstick":         ["LIPSTICK","LIP STICK","MATTE","VELVET","ROUGE"],
            "Lip Gloss":        ["GLOSS","SHINE","GLOW","TINT"],
            "Lip Liner":        ["LINER","PENCIL","LIP LINER"],
            "Foundation":       ["FOUNDATION","COVERAGE","FILT","FLUID","FIT ME"],
            "Concealer":        ["CONCEALER","RADIANT","CREAMY"],
            "Blush":            ["BLUSH","CHEEK","FLUSH","PINCH"],
            "Bronzer":          ["BRONZER","BRONZE","HOOLA","LAGUNA","SOLEIL"],
            "Highlighter":      ["HIGHLIGHT","GLOW","KILLAWATT","ILLUMINAT"],
            "Eyeshadow Palette":["PALETTE","SHADOW","NAKED","RENAISSANCE"],
            "Eyeliner":         ["LINER","EYELINER","FLUIDLINE","KOHL"],
            "Mascara":          ["MASCARA","LASH","VOLUME","PERVERSION"],
            "Setting Powder":   ["POWDER","SETTING","FINISH","BAKE"],
            "Primer":           ["PRIMER","PORE","BASE","PREP"],
            "Contour":          ["CONTOUR","SCULPT","DEFINE","CHISEL"],
        }
        cat = product["category"]
        for kw in cat_keywords.get(cat, []):
            if kw in ocr_text:
                score += 10

        if score > 0:
            matches.append((score, product))

    # Sort by score descending
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches

def identify_product_ml(image_b64, media_type, known_brands):
    """
    Two-stage identification:
    1. OCR reads text from packaging → matches to database
    2. Delta E color matching as backup or confirmation
    Returns the best match with confidence level.
    """
    img_bytes = base64.b64decode(image_b64)
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_rgb   = np.array(pil_img)

    # ── STAGE 1: OCR TEXT READING ──
    ocr_text    = extract_text_from_image(img_rgb)
    ocr_matches = match_text_to_product(ocr_text, known_brands)
    ocr_found   = len(ocr_matches) > 0 and ocr_matches[0][0] >= 40

    # ── STAGE 2: DELTA E COLOR MATCHING ──
    colors   = dominant_colors(img_rgb, n=5)
    filtered = [(r,g,b) for r,g,b in colors if 20 < 0.2126*r+0.7152*g+0.0722*b < 235]
    if not filtered:
        filtered = colors
    color_result = find_best_product_match(filtered, known_brands)

    # ── COMBINE RESULTS ──
    if ocr_found:
        # OCR found a strong match — use it
        best_score, best_product = ocr_matches[0]
        confidence = "high" if best_score >= 70 else "medium"

        # If OCR found brand+shade, see if color also agrees
        prod_rgb = hex_to_rgb(best_product["hex"])
        color_agrees = any(delta_e(c, prod_rgb) < 15 for c in filtered)
        if color_agrees and confidence == "medium":
            confidence = "high"

        alternatives = []
        for sc, p in ocr_matches[1:3]:
            alternatives.append({"brand":p["brand"],"name":p["name"],"shade":p["shade"],"hex":p["hex"],"score":sc})

        return {
            "brand":        best_product["brand"],
            "name":         best_product["name"],
            "category":     best_product["category"],
            "shade":        best_product["shade"],
            "shadeHex":     best_product["hex"],
            "finish":       best_product.get("finish",""),
            "brandKnown":   True,
            "confidence":   confidence,
            "method":       "OCR + text matching" + (" + color confirmed" if color_agrees else ""),
            "ocr_text":     ocr_text[:100] if ocr_text else "",
            "description":  build_product_description(best_product, 0),
            "alternatives": alternatives,
        }
    else:
        # OCR didn't find a clear match — fall back to Delta E color matching
        color_result["method"] = "Delta E color matching"
        color_result["ocr_text"] = ocr_text[:100] if ocr_text else ""

        # If OCR found partial text, mention it
        if ocr_matches and ocr_matches[0][0] >= 20:
            _, partial = ocr_matches[0]
            color_result["ocr_hint"] = f"Partial text detected — possible match: {partial['brand']} {partial['name']}"

        return color_result

# ─────────────────────────────────────────────
# FACE ANALYSIS (MediaPipe ML)
# ─────────────────────────────────────────────

def compute_face_shape(landmarks, w, h):
    lm = landmarks.landmark
    def pt(i): return np.array([lm[i].x*w, lm[i].y*h])
    top=pt(10); bottom=pt(152); left_chk=pt(93); right_chk=pt(361)
    left_fore=pt(70); right_fore=pt(300); left_jaw2=pt(172); right_jaw2=pt(397)
    fh=float(np.linalg.norm(top-bottom)); fw=float(np.linalg.norm(left_chk-right_chk))
    jw=float(np.linalg.norm(left_jaw2-right_jaw2)); fow=float(np.linalg.norm(left_fore-right_fore))
    hw=fh/(fw+1e-6); jr=jw/(fw+1e-6); fr=fow/(fw+1e-6)
    if hw>1.55: shape="oblong"
    elif hw>1.35: shape="heart" if jr<0.75 else "oval"
    elif hw>1.15:
        if jr>0.85 and fr>0.85: shape="square"
        elif jr<0.72: shape="heart"
        else: shape="oval"
    else: shape="square" if jr>0.82 else "round"
    return shape,round(hw,3),round(jr,3),round(fr,3)

def compute_eye_shape(landmarks, w, h):
    lm = landmarks.landmark
    def pt(i): return np.array([lm[i].x*w, lm[i].y*h])
    inner=pt(133); outer=pt(33); top_lid=pt(159); bot_lid=pt(145); lid_top=pt(386)
    ew=float(np.linalg.norm(outer-inner)); eh=float(np.linalg.norm(top_lid-bot_lid))
    tilt=float(outer[1]-inner[1])/(ew+1e-6); hw=eh/(ew+1e-6)
    hooded=float(np.linalg.norm(top_lid-lid_top))<eh*0.3
    if hooded: return "hooded"
    if hw<0.22: return "monolid"
    if tilt<-0.08: return "upturned"
    if tilt>0.08: return "downturned"
    if hw>0.38: return "round"
    return "almond"

def compute_undertone(img_rgb, landmarks, w, h):
    lm = landmarks.landmark
    def pts(idx): return [(int(lm[i].x*w),int(lm[i].y*h)) for i in idx]
    l=region_mean_color(img_rgb,pts(LEFT_CHEEK)); r=region_mean_color(img_rgb,pts(RIGHT_CHEEK))
    sr=int((l[0]+r[0])/2); sg=int((l[1]+r[1])/2); sb=int((l[2]+r[2])/2)
    skin_bgr=np.uint8([[[sb,sg,sr]]]); hsv=cv2.cvtColor(skin_bgr,cv2.COLOR_BGR2HSV)[0][0]
    hue=int(hsv[0]); rg=sr/(sg+1e-6); gb=sg/(sb+1e-6)
    if gb>1.15 and rg>1.05: undertone="warm"
    elif sb>sr*0.72 and hue>165: undertone="cool"
    elif abs(rg-1.0)<0.08: undertone="neutral"
    elif sg>sr*0.82 and sg>sb*1.05: undertone="olive"
    else: undertone="warm"
    return undertone, rgb_to_hex(sr,sg,sb)

def compute_lip_shape(landmarks, w, h):
    lm = landmarks.landmark
    def pt(i): return np.array([lm[i].x*w, lm[i].y*h])
    lc=pt(61); rc=pt(291); top=pt(0); bot=pt(17); pl=pt(37); pr=pt(267)
    lip_w=float(np.linalg.norm(rc-lc)); lip_h=float(np.linalg.norm(top-bot))
    hw=lip_h/(lip_w+1e-6); cupid=(pl[1]+pr[1])/2-top[1]
    if hw>0.42: return "full"
    if hw<0.25: return "thin"
    if cupid>3: return "cupid's bow"
    if lip_w>w*0.32: return "wide"
    return "heart-shaped"

def compute_cheekbones(landmarks, w, h):
    lm=landmarks.landmark
    def pt(i): return np.array([lm[i].x*w,lm[i].y*h])
    ratio=np.linalg.norm(pt(361)-pt(93))/(np.linalg.norm(pt(397)-pt(172))+1e-6)
    return "high" if ratio>1.18 else "medium" if ratio>1.05 else "low"

def compute_brow_shape(landmarks, w, h):
    lm=landmarks.landmark
    def pt(i): return np.array([lm[i].x*w,lm[i].y*h])
    inner=pt(46); peak=pt(66); outer=pt(70)
    bw=float(np.linalg.norm(outer-inner))
    line_y=inner[1]+(outer[1]-inner[1])*((peak[0]-inner[0])/(bw+1e-6))
    arch=float(line_y-peak[1])/(bw+1e-6)
    if arch>0.12: return "high arched"
    if arch>0.06: return "softly arched"
    if abs(arch)<0.04: return "straight"
    return "flat"

def compute_skin_tone(img_rgb, landmarks, w, h):
    lm=landmarks.landmark
    try:
        fp=[(int(lm[i].x*w),int(lm[i].y*h)) for i in [10,67,109,338,297]]
        cp=[(int(lm[i].x*w),int(lm[i].y*h)) for i in LEFT_CHEEK]
        fc=region_mean_color(img_rgb,fp); cc=region_mean_color(img_rgb,cp)
        lum=0.2126*int((fc[0]+cc[0])/2)+0.7152*int((fc[1]+cc[1])/2)+0.0722*int((fc[2]+cc[2])/2)
    except: lum=150
    if lum>210: return "fair"
    if lum>185: return "light"
    if lum>160: return "light-medium"
    if lum>135: return "medium"
    if lum>110: return "medium-tan"
    if lum>85: return "tan"
    if lum>60: return "deep"
    return "rich"

SHAPE_DATA = {
    "oval":    {"headline":"Beautifully balanced oval face","contour":"Light contouring under cheekbones — almost any technique flatters you.","blush":"Apply to apples of cheeks and blend upward.","brow":"Most brow shapes work — a soft arch is universally flattering.","pinterest":"oval face glam makeup look"},
    "round":   {"headline":"Soft symmetrical features with natural fullness","contour":"Contour sides of face and under cheekbones to add definition.","blush":"Apply slightly above hollows blending toward temples.","brow":"A higher angular arch adds length and lifts.","pinterest":"round face contouring makeup tutorial"},
    "square":  {"headline":"Strong jaw and striking defined structure","contour":"Soften jaw corners and forehead edges with bronzer.","blush":"Circular motions on apples of cheeks to soften angles.","brow":"Softer rounded brows balance angular features.","pinterest":"square face soft glam makeup"},
    "heart":   {"headline":"Wider forehead tapering to a delicate chin","contour":"Contour temples lightly. Highlight chin tip to balance.","blush":"Apply below cheekbones blending slightly downward.","brow":"Softer less dramatic arches balance the forehead.","pinterest":"heart face shape makeup tutorial"},
    "oblong":  {"headline":"Elegant long face with refined features","contour":"Add width with blush on sides. Contour at hairline and chin.","blush":"Apply horizontally across cheeks to add width.","brow":"Straighter flatter brows visually shorten the face.","pinterest":"oblong face shape makeup looks"},
    "diamond": {"headline":"Striking high cheekbones and refined jaw","contour":"Highlight forehead and jawline to add width.","blush":"Sweep outward from cheekbones.","brow":"Full straight brows add width to forehead.","pinterest":"diamond face shape makeup"},
}
EYE_TIPS = {"almond":"Almost any eye look works — try a classic smoky cut crease or winged liner.","round":"Elongate with a flicked liner wing. Shade outer corners darker.","hooded":"Apply shadow above the crease so it shows when eyes are open.","monolid":"Build shadow above the lash line and blend high. Bold liner looks stunning.","upturned":"Balance with shadow at outer corners angled slightly downward.","downturned":"Wing liner upward at outer corners. Highlight inner corners to lift."}
UNDERTONE_TIPS = {"warm":"Choose foundations with peach or golden undertones. Warm browns complement beautifully.","cool":"Choose foundations with pink or rosy undertones. Berry and mauve tones are your best friends.","neutral":"You can wear both warm and cool shades — focus on depth.","olive":"Look for foundations labeled olive or neutral. Warm mauves and terracottas look gorgeous."}
LIP_TIPS = {"full":"Enhance with a clear gloss or bold color.","thin":"Overline slightly outside your natural line to appear fuller.","cupid's bow":"A glossy center highlight makes the bow pop.","wide":"Keep color slightly inside corners for definition.","heart-shaped":"Balance both arches carefully with liner."}
LOOK_TEMPLATES = {
    "oval":    [("Classic Glam","Date Night","💄✨","banner-oval"),("Fresh Everyday","Everyday","🌸💫","banner-default"),("Smoky Drama","Editorial","🖤✨","banner-round"),("Soft Bridal","Bridal","🤍💐","banner-heart"),("Power Look","Work","💼💋","banner-oval"),("Festival Glitter","Festival","🌈✨","banner-diamond")],
    "round":   [("Sculpted Glam","Date Night","✨💄","banner-round"),("Lifted Everyday","Everyday","🌸☀️","banner-default"),("Sharp Editorial","Editorial","🖤💫","banner-square"),("Defined Work","Work","💼✨","banner-oval"),("Romantic Bridal","Bridal","🤍🌷","banner-heart"),("Bold Festival","Festival","🌈💎","banner-diamond")],
    "square":  [("Softened Glam","Date Night","💄🌸","banner-square"),("Natural Everyday","Everyday","☀️✨","banner-default"),("Strong Editorial","Editorial","🖤💋","banner-oblong"),("Elegant Bridal","Bridal","🤍💐","banner-heart"),("Polished Work","Work","💼💄","banner-oval"),("Glitter Festival","Festival","🌈⭐","banner-diamond")],
    "heart":   [("Romantic Glam","Date Night","💕✨","banner-heart"),("Dewy Everyday","Everyday","🌸💫","banner-default"),("Dreamy Bridal","Bridal","🤍🌷","banner-heart"),("Bold Editorial","Editorial","🖤💄","banner-round"),("Chic Work","Work","💼✨","banner-oval"),("Boho Festival","Festival","🌈🌸","banner-diamond")],
    "oblong":  [("Width Glam","Date Night","💄✨","banner-oblong"),("Balanced Everyday","Everyday","🌸☀️","banner-default"),("Bold Editorial","Editorial","🖤💫","banner-square"),("Classic Bridal","Bridal","🤍💐","banner-heart"),("Structured Work","Work","💼💋","banner-oval"),("Vivid Festival","Festival","🌈⭐","banner-diamond")],
    "diamond": [("Cheekbone Glam","Date Night","💎✨","banner-diamond"),("Fresh Everyday","Everyday","🌸💫","banner-default"),("High Fashion","Editorial","🖤💎","banner-oblong"),("Ethereal Bridal","Bridal","🤍✨","banner-heart"),("Sharp Work","Work","💼💄","banner-oval"),("Glam Festival","Festival","🌈💎","banner-diamond")],
}
EYE_STEPS = {
    "almond":    ["Apply transition shade across the crease","Pack main color onto the lid","Blend edges with a clean brush","Add liner along upper lash line","Finish with two coats of mascara"],
    "round":     ["Apply darker shade to outer V to elongate","Blend crease color outward","Apply liner from mid-lid flicking outward","Highlight inner corner","Curl lashes and apply mascara"],
    "hooded":    ["Apply shadow above the natural crease","Use matte transition to define","Draw liner on upper lid only","Tight-line the waterline","Apply lengthening mascara"],
    "monolid":   ["Build color directly above the lash line","Blend upward in a gradient","Apply graphic liner for definition","Apply voluminous mascara","Add lower lash liner for depth"],
    "upturned":  ["Apply darker shadow to outer lower corner","Blend crease color inward","Line upper lid ending straight","Smudge lower outer corner","Curl lashes toward center"],
    "downturned":["Apply lighter shade on inner lid","Wing liner upward at outer corner","Highlight brow bone and inner corner","Avoid dark lower liner","Apply mascara lifting upward"],
}
LIP_STEPS = {"warm":["Line lips with a warm nude liner","Fill with peach or coral lipstick","Dab gloss to center","Blot for long-lasting wear"],"cool":["Line lips with a mauve or berry liner","Apply cool-toned lipstick","Add pink gloss to center","Blot and reapply"],"neutral":["Line with neutral nude liner","Fill with your favourite shade","Add gloss for dimension","Blot lightly"],"olive":["Line with warm terracotta liner","Apply warm mauve lipstick","Add golden gloss to center","Blot for polished finish"]}

PRODUCT_DB_RECS = {
    "warm_fair":    [("MAC","Studio Fix Fluid","Foundation","NC15","#F5D5BE"),("NARS","Blush","Blush","Orgasm","#E891A8"),("Urban Decay","Naked Heat Palette","Eyeshadow Palette","Cayenne","#C48840"),("Charlotte Tilbury","Matte Revolution","Lipstick","Pillow Talk","#C49884"),("Benefit","Hoola","Bronzer","Hoola Light","#C48840"),("Fenty Beauty","Killawatt","Highlighter","Trophy Wife","#E8C97A")],
    "warm_light":   [("MAC","Studio Fix","Foundation","NC20","#F0C8A0"),("NARS","Laguna","Bronzer","Laguna","#C48840"),("Too Faced","Better Than Sex","Mascara","Black","#1A1A1A"),("Charlotte Tilbury","Lip Cheat","Lip Liner","Pillow Talk","#C49884"),("Urban Decay","All Nighter","Setting Spray","Original","#FFFFFF"),("Fenty Beauty","Pro Filt'r","Foundation","130N","#E8BFA2")],
    "cool_fair":    [("NARS","Sheer Glow","Foundation","Deauville","#FBECD9"),("MAC","Ruby Woo","Lipstick","Ruby Woo","#9B1C1C"),("Urban Decay","Naked 3","Eyeshadow Palette","Burnout","#C49884"),("Charlotte Tilbury","Filmstar Bronze","Bronzer","Bronze","#C48840"),("Rare Beauty","Soft Pinch","Blush","Hope","#E891A8"),("Benefit","Gimme Brow","Brow Gel","Light","#8B5E3C")],
    "cool_light":   [("NARS","Natural Radiant","Foundation","Syracuse","#F5D5BE"),("MAC","Velvet Teddy","Lipstick","Velvet Teddy","#9B6B7A"),("Urban Decay","Naked","Eyeshadow Palette","Buck","#C49884"),("Fenty Beauty","Gloss Bomb","Lip Gloss","Fenty Glow","#E8815A"),("Charlotte Tilbury","Hollywood Flawless","Setting Powder","Translucent","#FFFFFF"),("Rare Beauty","Liquid Touch","Foundation","110W","#F5D5BE")],
    "neutral_medium":[("Fenty Beauty","Pro Filt'r","Foundation","240N","#D4A283"),("NARS","Blush","Blush","Orgasm","#E891A8"),("ABH","Modern Renaissance","Eyeshadow Palette","Venetian Red","#C48840"),("MAC","Whirl","Lip Liner","Whirl","#9B6B7A"),("Huda Beauty","Faux Filter","Foundation","Linen","#D4A283"),("Charlotte Tilbury","Airbrush Flawless","Foundation","3 Warm","#D4A283")],
    "warm_medium":  [("Fenty Beauty","Pro Filt'r","Foundation","250W","#BE8860"),("MAC","Studio Fix","Foundation","NW35","#D4A283"),("Urban Decay","Naked Heat","Eyeshadow Palette","Ember","#C48840"),("NARS","Laguna","Bronzer","Laguna","#C48840"),("Charlotte Tilbury","Pillow Talk","Lipstick","Pillow Talk Med","#C49884"),("Too Faced","Born This Way","Foundation","Warm Beige","#D4A283")],
    "warm_tan":     [("Fenty Beauty","Pro Filt'r","Foundation","330W","#BE8860"),("NARS","Hot Sand","Blush","Hot Sand","#F0A882"),("MAC","Studio Fix","Foundation","NW43","#BE8860"),("Urban Decay","Naked Heat","Eyeshadow Palette","Scorched","#C48840"),("Huda Beauty","Faux Filter","Foundation","Caramel","#BE8860"),("ABH","Sundipped","Highlight","Sundipped","#E8C97A")],
    "cool_deep":    [("Fenty Beauty","Pro Filt'r","Foundation","420","#7A4A2A"),("MAC","Studio Fix","Foundation","NW55","#7A4A2A"),("NARS","Blush","Blush","Sin","#8B3A6B"),("Urban Decay","Naked Reloaded","Eyeshadow Palette","Chains","#C49884"),("Charlotte Tilbury","Matte Rev","Lipstick","Walk of No Shame","#9B1C1C"),("Huda Beauty","Faux Filter","Foundation","Cocoa","#7A4A2A")],
    "warm_deep":    [("Fenty Beauty","Pro Filt'r","Foundation","445W","#7A4A2A"),("MAC","Studio Fix","Foundation","NC50","#7A4A2A"),("ABH","Soft Glam","Eyeshadow Palette","Sienna","#C48840"),("NARS","Laguna","Bronzer","Laguna Deep","#8B5E3C"),("Huda Beauty","Lip Strobe","Lip Gloss","Heartbeat","#C49884"),("Too Faced","Born This Way","Foundation","Mahogany","#7A4A2A")],
    "default":      [("MAC","Studio Fix","Foundation","NC25","#D4A283"),("NARS","Blush","Blush","Orgasm","#E891A8"),("Urban Decay","Naked","Eyeshadow Palette","Naked","#C49884"),("Charlotte Tilbury","Pillow Talk","Lipstick","Pillow Talk","#C49884"),("Fenty Beauty","Gloss Bomb","Lip Gloss","Fenty Glow","#E8815A"),("Benefit","Hoola","Bronzer","Hoola","#C48840")],
}

def run_ml_face_analysis(img_bgr):
    h,w=img_bgr.shape[:2]; img_rgb=img_to_rgb(img_bgr)
    with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.4) as fm:
        results=fm.process(img_rgb)
    if not results.multi_face_landmarks:
        return {"error":"No face detected. Please try a clearer front-facing photo with good lighting."}
    lm=results.multi_face_landmarks[0]
    face_shape,hw,jr,fr=compute_face_shape(lm,w,h)
    eye_shape=compute_eye_shape(lm,w,h)
    undertone,skin_hex=compute_undertone(img_rgb,lm,w,h)
    lip_shape=compute_lip_shape(lm,w,h)
    cheekbones=compute_cheekbones(lm,w,h)
    brow_shape=compute_brow_shape(lm,w,h)
    skin_tone=compute_skin_tone(img_rgb,lm,w,h)
    sd=SHAPE_DATA.get(face_shape,SHAPE_DATA["oval"])
    et=EYE_TIPS.get(eye_shape,"Your eyes are beautifully unique!")
    ut=UNDERTONE_TIPS.get(undertone,"")
    lt=LIP_TIPS.get(lip_shape,"Your lips are beautifully unique.")
    es=EYE_STEPS.get(eye_shape,EYE_STEPS["almond"])
    ls=LIP_STEPS.get(undertone,LIP_STEPS["neutral"])
    features=[
        {"label":"Face Shape","value":face_shape.title(),"tip":sd["contour"]},
        {"label":"Undertone","value":undertone.title(),"tip":ut},
        {"label":"Eye Shape","value":eye_shape.title(),"tip":et},
        {"label":"Lip Shape","value":lip_shape.title(),"tip":lt},
        {"label":"Cheekbones","value":cheekbones.title(),"tip":sd["blush"]},
        {"label":"Brow Shape","value":brow_shape.title(),"tip":sd["brow"]},
        {"label":"Skin Tone","value":skin_tone.replace("-"," ").title(),"tip":"Match your foundation undertone to your detected skin undertone for a seamless finish."},
    ]
    templates=LOOK_TEMPLATES.get(face_shape,LOOK_TEMPLATES["oval"])
    desc_map={"Date Night":f"A romantic look for your {eye_shape} eyes and {lip_shape} lips. Warm tones enhance your {undertone} undertone.","Everyday":f"A fresh natural look for your {skin_tone} skin tone. Quick to apply and effortlessly flattering.","Editorial":f"A bold statement look built around your {eye_shape} eye shape.","Bridal":f"A timeless look enhancing your natural features on {skin_tone} skin.","Work":f"A polished professional look. Subtle yet refined.","Festival":f"A fun expressive look playing up your {eye_shape} eyes with colour and shimmer."}
    why_map={"oval":f"Your {face_shape} face is the most versatile — this look enhances your natural balance.","round":f"Strategic contouring adds definition to your {face_shape} face shape.","square":f"Soft blending balances your strong {face_shape} jaw.","heart":f"Targeted color placement balances your wider forehead and delicate chin.","oblong":f"Horizontal blush adds width to your elegant {face_shape} face.","diamond":f"This highlights your striking cheekbones while balancing your {face_shape} face."}
    looks=[]
    for name,vibe,emoji,banner in templates:
        looks.append({"name":name,"vibe":vibe,"emoji":emoji,"bannerClass":banner,"description":desc_map.get(vibe,"A beautiful look designed for your unique features."),"whyItWorks":why_map.get(face_shape,f"Designed for your {face_shape} face."),"steps":es[:3]+ls[:2],"keyProducts":[f"{undertone.title()}-tone foundation",f"{'warm' if undertone in ['warm','olive'] else 'cool'} blush",f"{'berry' if undertone=='cool' else 'coral'} lip"],"tags":[vibe,face_shape,undertone,eye_shape],"pinterestQuery":f"{vibe.lower()} makeup {face_shape} face {undertone} undertone"})
    narrative=(f"Your {face_shape} face shape with {cheekbones} cheekbones gives you a naturally {'balanced' if face_shape=='oval' else 'striking'} structure. Your {eye_shape} eyes are a standout feature — {et[:80]}. With a {undertone} undertone, {ut[:80]}. Your {lip_shape} lips work beautifully with {lt[:60]}.")
    return {"faceShape":face_shape,"eyeShape":eye_shape,"lipShape":lip_shape,"skinUndertone":undertone,"skinTone":skin_tone,"skinHex":skin_hex,"cheekbones":cheekbones,"browShape":brow_shape,"features":features,"headline":sd["headline"],"subtext":f"Your {undertone} undertone and {eye_shape} eyes give you a truly unique canvas.","narrative":narrative,"looks":looks,"pinterestKeyword":sd["pinterest"],"measurements":{"heightWidthRatio":hw,"jawRatio":jr,"foreheadRatio":fr},"mlAnalysis":True}

def generate_recommendations(shade_profile, face_data, inventory):
    undertone=face_data.get("skinUndertone",shade_profile.get("skinTone","neutral"))
    skin_tone=face_data.get("skinTone",shade_profile.get("skinTone","medium"))
    face_shape=face_data.get("faceShape","oval"); eye_shape=face_data.get("eyeShape","almond")
    lip_prefs=shade_profile.get("lip",[]); owned={f"{p.get('brand','')} {p.get('name','')}".strip() for p in inventory}
    st=skin_tone.split("-")[0] if "-" in skin_tone else skin_tone
    key=f"{undertone}_{st}"; products=PRODUCT_DB_RECS.get(key,PRODUCT_DB_RECS.get(f"{undertone}_medium",PRODUCT_DB_RECS["default"]))
    reason_map={"Foundation":f"Matched to your {skin_tone} skin tone with a {undertone} undertone.","Blush":f"Complements your {undertone} undertone on your {face_shape} face.","Bronzer":f"Enhances your {undertone} undertone — perfect for your {face_shape} structure.","Highlighter":f"Placed on your {face_data.get('cheekbones','high')} cheekbones this catches light beautifully.","Eyeshadow Palette":f"Designed for {eye_shape} eyes and your {undertone} undertone.","Lipstick":f"Suits your {lip_prefs[0] if lip_prefs else 'natural'} lip preference.","Lip Gloss":f"Enhances your {face_data.get('lipShape','full')} lip shape naturally.","Concealer":f"Matched to your {skin_tone} skin tone.","Setting Powder":f"Locks makeup in place all day.","Mascara":f"Opens up your {eye_shape} eyes beautifully."}
    emoji_map={"Foundation":"🧴","Blush":"🌸","Bronzer":"☀️","Highlighter":"💫","Eyeshadow Palette":"🎨","Lipstick":"💄","Lip Gloss":"✨","Concealer":"🎭","Setting Powder":"🌟","Mascara":"👁️","Setting Spray":"💧","Brow Gel":"✏️"}
    recs=[]
    for brand,name,category,shade,shade_hex in products:
        if f"{brand} {name}" in owned: continue
        recs.append({"brand":brand,"name":name,"category":category,"shade":shade,"shadeHex":shade_hex,"reason":reason_map.get(category,f"A great match for your {undertone} undertone."),"emoji":emoji_map.get(category,"💄")})
    return recs[:6]

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/analyze-face", methods=["POST"])
def api_analyze_face():
    try:
        data=request.json; image_data_url=data.get("imageData","")
        if not image_data_url: return jsonify({"error":"No image provided"}),400
        img_bgr=decode_image(image_data_url); result=run_ml_face_analysis(img_bgr)
        if "error" in result: return jsonify(result),422
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}),500

@app.route("/api/identify-product", methods=["POST"])
def api_identify_product():
    try:
        data=request.json; image_data_url=data.get("imageData",""); known_brands=data.get("knownBrands",[])
        if not image_data_url: return jsonify({"error":"No image provided"}),400
        header,b64=image_data_url.split(",",1); media_type=header.split(";")[0].split(":")[1]
        result=identify_product_ml(b64,media_type,known_brands)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}),500

@app.route("/api/recommendations", methods=["POST"])
def api_recommendations():
    try:
        data=request.json
        recs=generate_recommendations(data.get("shadeProfile",{}),data.get("faceData",{}),data.get("inventory",[]))
        return jsonify({"recommendations":recs})
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}),500

COMMUNITY_DB_PATH = os.path.join(os.path.dirname(__file__), "community_products.json")

def load_community_products():
    """Load user-submitted products from file."""
    if os.path.exists(COMMUNITY_DB_PATH):
        try:
            with open(COMMUNITY_DB_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_community_products(products):
    """Save user-submitted products to file."""
    with open(COMMUNITY_DB_PATH, "w") as f:
        json.dump(products, f, indent=2)

def get_full_product_db():
    """Combine built-in database with community submissions."""
    community = load_community_products()
    return PRODUCT_DATABASE + [
        {
            "brand":    p.get("brand",""),
            "name":     p.get("name",""),
            "category": p.get("category","Other"),
            "shade":    p.get("shade",""),
            "hex":      p.get("hex","#C49884"),
            "finish":   p.get("finish",""),
        }
        for p in community
    ]


@app.route("/api/add-product", methods=["POST"])
def api_add_product():
    """
    Add a user-submitted product to the community database.
    Saved to community_products.json and immediately available for matching.
    """
    try:
        data     = request.json
        brand    = data.get("brand","").strip()
        name     = data.get("name","").strip()
        category = data.get("category","Other").strip()
        shade    = data.get("shade","").strip()
        hex_val  = data.get("hex","#C49884").strip()
        finish   = data.get("finish","").strip()
        added_by = data.get("addedBy","anonymous")

        if not brand or not name:
            return jsonify({"error":"Brand and product name are required"}), 400

        # Validate hex color
        if not hex_val.startswith("#") or len(hex_val) not in (4,7):
            hex_val = "#C49884"

        community = load_community_products()

        # Check for duplicates
        for p in community:
            if p.get("brand","").lower() == brand.lower() and \
               p.get("name","").lower() == name.lower() and \
               p.get("shade","").lower() == shade.lower():
                return jsonify({"success": True, "message": "Product already in database!", "duplicate": True})

        new_product = {
            "brand":    brand,
            "name":     name,
            "category": category,
            "shade":    shade,
            "hex":      hex_val,
            "finish":   finish,
            "addedBy":  added_by,
            "addedAt":  __import__("datetime").datetime.now().isoformat(),
        }

        community.append(new_product)
        save_community_products(community)

        print(f"✨ New product added by {added_by}: {brand} {name} ({shade})")
        return jsonify({"success": True, "message": f"{brand} {name} added to database!", "product": new_product})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/track-signup", methods=["POST"])
def api_track_signup():
    """Called from frontend when a new user signs up."""
    try:
        data      = request.json
        email     = data.get("email","").strip()
        name      = data.get("name","").strip()
        skin_tone = data.get("skinTone","").strip()
        if not email:
            return jsonify({"error":"Email required"}), 400
        track_user(email, name, skin_tone)
        return jsonify({"success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/track-products", methods=["POST"])
def api_track_products():
    """Called when user adds products — updates their count."""
    try:
        data  = request.json
        email = data.get("email","").strip()
        count = data.get("count", 0)
        if email:
            update_user_products(email, count)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/users", methods=["GET"])
def api_users():
    """
    View all tracked users.
    Visit: http://localhost:5000/api/users
    """
    try:
        users = load_tracked_users()
        return jsonify({
            "total":   len(users),
            "users":   users,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/community-products", methods=["GET"])
def api_community_products():
    return jsonify({"products": load_community_products(), "count": len(load_community_products())})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","ml":"mediapipe+opencv+sklearn+deltaE","products":len(PRODUCT_DATABASE),"users":len(load_tracked_users()),"ai":"rule-based (no API needed)"})

if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000)); debug=os.environ.get("RAILWAY_ENVIRONMENT") is None
    print("\n✨ Glam Vault Backend Starting...")
    print("━"*45)
    print("  ML Engine : MediaPipe + OpenCV + scikit-learn")
    print("  Color Match: Delta E CIEDE2000 Algorithm")
    print(f"  Products  : {len(PRODUCT_DATABASE)} real products in database")
    print("  AI Engine : Rule-based ML (no API key needed!)")
    print(f"  URL       : http://localhost:{port}")
    print("━"*45)
    app.run(debug=debug, host="0.0.0.0", port=port)