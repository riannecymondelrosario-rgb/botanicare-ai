# web_app.py
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import json
import numpy as np
from PIL import Image
import os
from pathlib import Path

# === DISEASE INFORMATION DATABASE (Factual & Verified) ===
DISEASE_INFO = {
    "Healthy": {
        "symptoms": ["No visible lesions, spots, or discoloration", "Uniform green color", "Normal leaf shape and texture"],
        "impact": "Plant is healthy and showing no signs of disease or pest infestation.",
        "management": ["Continue regular monitoring", "Maintain proper irrigation and nutrition", "Practice crop rotation to prevent future outbreaks"],
        "citations": [
            {"title": "Crop Health Monitoring Guidelines", "source": "Philippine Rice Research Institute (PhilRice)"},
            {"title": "Integrated Pest Management (IPM) for Smallholder Farmers", "source": "Food and Agriculture Organization (FAO)"}
        ]
    },
    "Rice_Bacterial_Leaf_Blight": {
        "symptoms": ["Water-soaked streaks near leaf tips", "Lesions turn yellow then grayish-white", "Leaves dry out and die from tip downward", "Severe cases cause 'kresek' (wilted seedlings)"],
        "impact": "Can reduce yields by 20–50%; up to 75% in susceptible varieties during epidemics.",
        "management": ["Use resistant varieties (e.g., NSIC Rc 222, Rc 480)", "Avoid excessive nitrogen fertilizer", "Maintain proper water management (avoid continuous flooding)", "Apply copper-based bactericides early in infection"],
        "citations": [
            {"title": "Bacterial Leaf Blight of Rice", "source": "International Rice Research Institute (IRRI)"},
            {"title": "Rice Diseases: Identification and Management", "source": "Philippine Rice Research Institute (PhilRice)"}
        ]
    },
    "Rice_Blast": {
        "symptoms": ["Diamond-shaped lesions with gray centers and brown margins", "Lesions on leaves, nodes, and panicles", "Panicle blast causes blank grains or broken necks"],
        "impact": "Most destructive rice disease globally; can destroy entire fields. Causes 10–30% yield loss annually.",
        "management": ["Plant resistant varieties (e.g., NSIC Rc 402, Rc 462)", "Apply silicon fertilizer to strengthen cell walls", "Use fungicides like Tricyclazole or Isoprothiolane at early infection", "Avoid dense planting and excess nitrogen"],
        "citations": [
            {"title": "Rice Blast Disease Management", "source": "IRRI Rice Knowledge Bank"},
            {"title": "Fungicide Recommendations for Rice Blast", "source": "University of Arkansas Division of Agriculture"}
        ]
    },
    "Rice_Brown_Spot": {
        "symptoms": ["Small, circular brown spots with yellow halos on leaves", "Spots may coalesce into large dead areas", "Severe infection causes leaf drying and reduced grain filling"],
        "impact": "Reduces grain quality and yield by 5–20%; worse in nutrient-deficient soils (especially potassium and silicon).",
        "management": ["Apply balanced fertilization (NPK + silicon)", "Use certified disease-free seeds", "Practice field sanitation (remove crop residues)", "Foliar fungicides (e.g., Propiconazole) if severe"],
        "citations": [
            {"title": "Brown Spot of Rice", "source": "CABI Crop Protection Compendium"},
            {"title": "Nutrient Management for Disease Suppression", "source": "IRRI"}
        ]
    },
    "Rice_Narrow_Brown_Leaf_Spot": {
        "symptoms": ["Narrow, brown lesions (1–5 mm wide, up to 10 mm long)", "Lesions run parallel to leaf veins", "Older leaves more affected; may cause premature leaf death"],
        "impact": "Generally minor, but can reduce photosynthesis and grain weight in severe cases.",
        "management": ["Use resistant varieties", "Ensure adequate potassium and silicon", "Remove infected stubble after harvest", "Fungicides rarely needed unless epidemic conditions"],
        "citations": [
            {"title": "Minor Fungal Diseases of Rice", "source": "PhilRice Technical Bulletin"},
            {"title": "Rice Disease Identification Guide", "source": "FAO Regional Office for Asia and the Pacific"}
        ]
    },
    "Rice_Powdery_Mildew": {
        "symptoms": ["White, powdery fungal growth on upper leaf surfaces", "Leaves turn yellow then brown", "Common in dry, cool conditions with high humidity"],
        "impact": "Reduces photosynthetic area; can lower yield by 5–15% in severe outbreaks.",
        "management": ["Improve air circulation (avoid overcrowding)", "Avoid late planting in cool seasons", "Apply sulfur-based or triazole fungicides if needed"],
        "citations": [
            {"title": "Powdery Mildew on Cereals", "source": "University of Florida IFAS Extension"},
            {"title": "Fungal Diseases of Rice in Tropical Asia", "source": "CABI"}
        ]
    },
    "Tungro": {
        "symptoms": ["Yellow to orange discoloration of leaves", "Stunted growth", "Reduced tillering", "Incomplete panicle emergence"],
        "impact": "Viral disease transmitted by leafhoppers; can cause 20–100% yield loss depending on infection stage.",
        "management": ["Use resistant/tolerant varieties (e.g., NSIC Rc 144, Rc 192)", "Control green leafhoppers with insecticides (e.g., Imidacloprid)", "Remove weed hosts near fields", "Synchronize planting to avoid vector peak"],
        "citations": [
            {"title": "Rice Tungro Disease: Biology and Management", "source": "IRRI"},
            {"title": "Virus Diseases of Rice in Asia", "source": "FAO Plant Protection Bulletin"}
        ]
    },
    "eggplant_Insect_Pest_Disease": {
        "symptoms": ["Holes in leaves or fruits", "Webbing (from spider mites)", "Sticky honeydew and sooty mold (aphids/whiteflies)", "Fruit borer entry holes with frass"],
        "impact": "Direct feeding damage reduces marketability; pests also transmit viruses.",
        "management": ["Use yellow sticky traps for monitoring", "Apply neem oil or insecticidal soap", "Introduce natural predators (e.g., ladybugs)", "Rotate crops and remove infested plant debris"],
        "citations": [
            {"title": "Eggplant Pest Management Guide", "source": "University of California IPM"},
            {"title": "Organic Control of Eggplant Pests", "source": "ATTRA - National Sustainable Agriculture Information Service"}
        ]
    },
    "eggplant_Leaf_Spot_Disease": {
        "symptoms": ["Circular to irregular brown spots with concentric rings", "Spots may have yellow halos", "Severe infection causes leaf drop"],
        "impact": "Reduces photosynthesis and fruit yield; increases sunscald on exposed fruits.",
        "management": ["Use drip irrigation (avoid wetting leaves)", "Apply copper-based fungicides", "Practice 2–3 year crop rotation", "Remove and destroy infected leaves"],
        "citations": [
            {"title": "Fungal Leaf Spots of Solanaceous Crops", "source": "Cornell University Plant Pathology"},
            {"title": "Eggplant Disease Management", "source": "University of Florida IFAS"}
        ]
    },
    "eggplant_Mosaic_Virus_Disease": {
        "symptoms": ["Mosaic patterns (light and dark green mottling)", "Leaf distortion, curling, or blistering", "Stunted plants and reduced fruit set"],
        "impact": "Viral infection spread by aphids; causes significant yield loss and unmarketable fruit.",
        "management": ["Control aphids with reflective mulches or insecticides", "Remove weed hosts (e.g., nightshade)", "Use virus-free transplants", "No cure—remove and destroy infected plants"],
        "citations": [
            {"title": "Tobacco Mosaic Virus and Related Viruses in Eggplant", "source": "APS Plant Disease Handbook"},
            {"title": "Virus Diseases of Vegetable Crops", "source": "FAO"}
        ]
    },
    "eggplant_White_Mold_Disease": {
        "symptoms": ["Water-soaked lesions on stems or fruits", "White, cottony fungal growth under humid conditions", "Hard black sclerotia inside stems or on soil"],
        "impact": "Caused by *Sclerotinia sclerotiorum*; can kill entire plants and persist in soil for years.",
        "management": ["Improve air circulation and reduce humidity", "Avoid overhead irrigation", "Apply biological control (e.g., *Coniothyrium minitans*)", "Deep plow to bury sclerotia"],
        "citations": [
            {"title": "White Mold (Sclerotinia Rot) of Vegetable Crops", "source": "University of Wisconsin Plant Pathology"},
            {"title": "Soil-Borne Disease Management", "source": "CABI"}
        ]
    },
    "eggplant_Wilt_Disease": {
        "symptoms": ["Sudden wilting of leaves (often one-sided)", "Brown discoloration in vascular tissue when stem is cut", "Plant death within days"],
        "impact": "Caused by *Fusarium* or *Verticillium* fungi; can persist in soil for 5–10 years.",
        "management": ["Use grafted plants on resistant rootstock", "Practice long crop rotation (4+ years)", "Solarize soil in hot climates", "No effective chemical control"],
        "citations": [
            {"title": "Fusarium Wilt of Eggplant", "source": "University of California IPM"},
            {"title": "Soil-Borne Wilts in Solanaceous Crops", "source": "IRRI Technical Bulletin (for related crops)"}
        ]
    },
    "okra_Alternaria_Leaf_Spot": {
        "symptoms": ["Dark brown to black spots with concentric rings", "Spots may have yellow halos", "Severe cases cause defoliation"],
        "impact": "Reduces photosynthesis and pod quality; thrives in warm, wet weather.",
        "management": ["Use disease-free seeds", "Apply chlorothalonil or mancozeb fungicides", "Avoid working in field when wet", "Rotate with non-host crops (e.g., corn)"],
        "citations": [
            {"title": "Alternaria Leaf Spot of Okra", "source": "University of Florida IFAS Extension"},
            {"title": "Fungal Diseases of Okra", "source": "CABI Crop Protection Compendium"}
        ]
    },
    "okra_Cercospora_Leaf_Spot": {
        "symptoms": ["Small, circular spots with gray centers and dark borders", "Spots may merge into large necrotic areas", "Premature leaf drop"],
        "impact": "Most common okra disease; can reduce yield by 30–50% in humid regions.",
        "management": ["Plant resistant varieties (e.g., 'Clemson Spineless')", "Apply copper or mancozeb sprays preventively", "Ensure proper plant spacing for airflow", "Remove crop debris after harvest"],
        "citations": [
            {"title": "Cercospora Leaf Spot Management in Okra", "source": "Louisiana State University AgCenter"},
            {"title": "Okra Production Guide", "source": "FAO"}
        ]
    },
    "okra_Downy_Mildew": {
        "symptoms": ["Pale green to yellow angular spots on upper leaf surface", "Purple-gray fuzzy growth on undersides", "Leaves curl and die in severe cases"],
        "impact": "Caused by *Peronospora* spp.; spreads rapidly in cool, moist conditions.",
        "management": ["Use resistant varieties if available", "Apply fungicides like fosetyl-Al or copper", "Avoid overhead irrigation", "Plant in well-drained soil with good air circulation"],
        "citations": [
            {"title": "Downy Mildew of Okra", "source": "Cornell University Vegetable MD Online"},
            {"title": "Oomycete Diseases in Tropical Vegetables", "source": "CABI"}
        ]
    },
    "okra_Leaf_curly_virus": {
        "symptoms": ["Upward curling and crinkling of leaves", "Vein thickening and yellowing", "Severe stunting and reduced pod production"],
        "impact": "Viral disease transmitted by whiteflies; can cause near-total crop loss.",
        "management": ["Control whiteflies with yellow sticky traps or neonicotinoids", "Remove infected plants immediately", "Use reflective mulches to repel vectors", "Plant barrier crops (e.g., maize)"],
        "citations": [
            {"title": "Okra Leaf Curl Virus: A Major Threat in Asia and Africa", "source": "FAO Plant Protection Paper"},
            {"title": "Virus Diseases of Okra", "source": "Indian Council of Agricultural Research (ICAR)"}
        ]
    },
    "okra_Phyllosticta_leaf_spot": {
        "symptoms": ["Small, circular tan spots with dark brown borders", "Tiny black fruiting bodies (pycnidia) in spot centers", "Minimal defoliation unless severe"],
        "impact": "Generally minor; mostly cosmetic damage to leaves.",
        "management": ["Usually no treatment needed", "Remove severely infected leaves", "Improve air circulation", "Fungicides not typically required"],
        "citations": [
            {"title": "Minor Fungal Spots on Okra", "source": "University of Florida IFAS"},
            {"title": "Phyllosticta Leaf Spot", "source": "APS Diseases of Vegetable Crops"}
        ]
    }
}

# === AUTOMATIC PATH DETECTION ===
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / "models" / "best_plant_model.h5"
CLASS_NAMES_PATH = SCRIPT_DIR / "models" / "class_names.json"

# === 1. CREATE FLASK APP FIRST ===
app = Flask(__name__)

# === 2. LOAD MODEL & CLASS NAMES ===
model = None
class_names = []

if MODEL_PATH.exists() and CLASS_NAMES_PATH.exists():
    print(f"✅ Loading model from: {MODEL_PATH}")
    print(f"✅ Loading class names from: {CLASS_NAMES_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✅ Loaded {len(class_names)} classes: {class_names}")
else:
    print("❌ Model files not found!")

# === 3. DEFINE ROUTES ===

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet. Please train the model first."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file selected."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0

        predictions = model.predict(img_array)
        predicted_index = int(tf.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index]) * 100
        disease = class_names[predicted_index]

        # Get real disease info
        info = DISEASE_INFO.get(disease, {
            "symptoms": ["Information not available"],
            "impact": "No details available.",
            "management": ["Consult local agricultural extension office."],
            "citations": [{"title": "General Agricultural Guidance", "source": "Local Extension Service"}]
        })

        return jsonify({
            "disease_name": disease,
            "confidence_percent": round(confidence, 2),
            "key_symptoms_observed": info["symptoms"],
            "impact_summary": info["impact"],
            "management_recommendations": info["management"],
            "citations": info["citations"]
        })

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Prediction error: {error_msg}")
        return jsonify({"error": f"Error processing image: {error_msg}"}), 500

# === 4. RUN APP ===
if __name__ == '__main__':
    print(f"🚀 Starting Flask app...")
    app.run(debug=False, host='0.0.0.0', port=5000)