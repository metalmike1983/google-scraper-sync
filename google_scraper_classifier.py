import os
import re
import io
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from googlesearch import search
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
torch.classes.__path__ = []  # <== WORKAROUND PER BUG #8488
import torch.nn.functional as F
from deep_translator import GoogleTranslator
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import streamlit as st
import warnings
from dateutil.parser import parse
import fitz  # PyMuPDF for PDF processing
import tempfile
import subprocess
import tempfile
import subprocess
import random


warnings.filterwarnings("ignore", category=UserWarning)

# === CONFIG ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "mike-83/longformer_policy_classifier"
TEXT_SIM_THRESHOLD = 0.85
PDF_TEXT_LIMIT = 5000  # Max number of characters to extract from PDF

EXCLUDED_DOMAINS = [
    "soundcloud.com", "youtube.com", "youtu.be", "spotify.com",
    "vimeo.com", "facebook.com", "instagram.com"
]

FAPDA_TOPICS = [
   "Unspecified tax policy",
"Value-added tax (VAT)",
"Tax on fuel and water",
"Other indirect tax",
"Income tax",
"Unspecified social protection measures",
"In-kind food transfer",
"Food- for -work",
"School feeding",
"Soup kitchen and food pantries",
"Food coupons",
"Food subsidy",
"Unconditional cash transfer",
"Conditional cash transfer (CCT)",
"Cash-for-work",
"Subsidies on fuel, power and water",
"Unspecified market policy",
"Establishment or modification of food stock",
"Release of food stock",
"Price control",
"Institutional reform measure",
"Legal and regulatory measures for consumer protection",
"Food safety regulations and standards",
"Unspecified disposable income policy",
"Salaries of civil servants",
"Minimum wages",
"Credit for consumption",
"Unemployment compensation",
"Employment programmes",
"Nutrition and health policy",
"Food fortification",
"Interventions to improve intake/absorption of micronutrients",
"Breastfeeding promotion",
"Therapeutic feeding",
"Public awareness and dietary practices",
"Drinking water",
"Sanitation and hygiene",
"Unspecified production support",
"General input measures",
"Fertiliser subsidies/vouchers",
"Fertiliser distribution",
"Local production of fertilisers and other agricultural inputs",
"Seed subsidies/vouchers",
"Seed distribution",
"Seed technology and quality assurance systems",
"Fuel resources for production",
"Machinery support (subsidies or distribution)",
"Livestock and livestock feed distribution",
"Unspecified agricultural tax",
"Tax on inputs or fixed capital",
"Farm income tax",
"Unspecified credit and finance facility",
"Access to credit",
"Financial support through public banks",
"Unspecified policy for knowledge generation and dissemination",
"Agriculture research and technology",
"Technical assistance, extension and training",
"Livestock measures and regulations",
"Fisheries and aquaculture measures and regulations",
"Production subsidies",
"Support to productive assets",
"Support to irrigation infrastructure",
"Unspecified genetic resources and sanitary measures",
"Animal genetic resources measures",
"Plant genetic resources measures",
"Animal health measures",
"Plant health measures",
"Food safety measures",
"Unspecified government market intervention",
"Price intervention on staple commodities",
"Price intervention on cash crop commodities",
"Government procurement from domestic farmers",
"Unspecified risk management measures",
"Marketing, production and derivative contracts",
"Insurance and reinsurance",
"Public/mutual fund and contingent risk financing",
"Unspecified value chain development measure",
"National market information system",
"Processing and post-production facilities",
"Transport regulation and infrastructure",
"Promotion of farmer markets or community markets",
"Unspecified measures for the management and conservation of natural resources",
"Water policies and regulations",
"Ecosystem and habitat preservation",
"Forest policies and regulations",
"Fisheries and aquaculture resources",
"Climate change mitigation and adaptation measures",
"Renewable energy and energy efficiency measures",
"Unspecified land policy measure",
"Land-use planning and land management",
"Land ownership, tenure and titling",
"Institutional measure",
"Public institution",
"Privatization",
"Institutional enforcement of producers organizations",
"Import tariff",
"Import ban",
"Import quota",
"Tariffrate quota",
"Other import restrictions",
"Import subsidy",
"Antidumping duties, countervailing duties, safeguard measures",
"Sanitary and Phytosanitary measures (SPS)",
"Technical barriers to trade",
"Other measures that affect imports",
"Export tax",
"Export ban",
"Export quota",
"Other export restrictions",
"Export subsidy",
"Sanitary, phytosanitary and technical standards improvements",
"Other export promotion measures",
"Other measures that affect exports",
"Competition policy",
"Government procurement through imports",
"Trade facilitation",
"Foreign exchange policy",
"Free or preferential trade agreement",
"Customs Union",
"Common market/Economic unions",
"Other trade and traderelated measures",
"Macroeconomic policy",
"Agricultural expenditure in the national budget"
]


default_topics =    ["Unspecified tax policy",
"Value-added tax (VAT)",
"Tax on fuel and water",
"Other indirect tax",
"Income tax",
"Unspecified social protection measures",
"In-kind food transfer",
"Food- for -work",
"School feeding",
"Soup kitchen and food pantries",
"Food coupons",
"Food subsidy",
"Unconditional cash transfer",
"Conditional cash transfer (CCT)",
"Cash-for-work",
"Subsidies on fuel, power and water",
"Unspecified market policy",
"Establishment or modification of food stock",
"Release of food stock",
"Price control",
"Institutional reform measure",
"Legal and regulatory measures for consumer protection",
"Food safety regulations and standards",
"Unspecified disposable income policy",
"Salaries of civil servants",
"Minimum wages",
"Credit for consumption",
"Unemployment compensation",
"Employment programmes",
"Nutrition and health policy",
"Food fortification",
"Interventions to improve intake/absorption of micronutrients",
"Breastfeeding promotion",
"Therapeutic feeding",
"Public awareness and dietary practices",
"Drinking water",
"Sanitation and hygiene",
"Unspecified production support",
"General input measures",
"Fertiliser subsidies/vouchers",
"Fertiliser distribution",
"Local production of fertilisers and other agricultural inputs",
"Seed subsidies/vouchers",
"Seed distribution",
"Seed technology and quality assurance systems",
"Fuel resources for production",
"Machinery support (subsidies or distribution)",
"Livestock and livestock feed distribution",
"Unspecified agricultural tax",
"Tax on inputs or fixed capital",
"Farm income tax",
"Unspecified credit and finance facility",
"Access to credit",
"Financial support through public banks",
"Unspecified policy for knowledge generation and dissemination",
"Agriculture research and technology",
"Technical assistance, extension and training",
"Livestock measures and regulations",
"Fisheries and aquaculture measures and regulations",
"Production subsidies",
"Support to productive assets",
"Support to irrigation infrastructure",
"Unspecified genetic resources and sanitary measures",
"Animal genetic resources measures",
"Plant genetic resources measures",
"Animal health measures",
"Plant health measures",
"Food safety measures",
"Unspecified government market intervention",
"Price intervention on staple commodities",
"Price intervention on cash crop commodities",
"Government procurement from domestic farmers",
"Unspecified risk management measures",
"Marketing, production and derivative contracts",
"Insurance and reinsurance",
"Public/mutual fund and contingent risk financing",
"Unspecified value chain development measure",
"National market information system",
"Processing and post-production facilities",
"Transport regulation and infrastructure",
"Promotion of farmer markets or community markets",
"Unspecified measures for the management and conservation of natural resources",
"Water policies and regulations",
"Ecosystem and habitat preservation",
"Forest policies and regulations",
"Fisheries and aquaculture resources",
"Climate change mitigation and adaptation measures",
"Renewable energy and energy efficiency measures",
"Unspecified land policy measure",
"Land-use planning and land management",
"Land ownership, tenure and titling",
"Institutional measure",
"Public institution",
"Privatization",
"Institutional enforcement of producers organizations",
"Import tariff",
"Import ban",
"Import quota",
"Tariffrate quota",
"Other import restrictions",
"Import subsidy",
"Antidumping duties, countervailing duties, safeguard measures",
"Sanitary and Phytosanitary measures (SPS)",
"Technical barriers to trade",
"Other measures that affect imports",
"Export tax",
"Export ban",
"Export quota",
"Other export restrictions",
"Export subsidy",
"Sanitary, phytosanitary and technical standards improvements",
"Other export promotion measures",
"Other measures that affect exports",
"Competition policy",
"Government procurement through imports",
"Trade facilitation",
"Foreign exchange policy",
"Free or preferential trade agreement",
"Customs Union",
"Common market/Economic unions",
"Other trade and traderelated measures",
"Macroeconomic policy",
"Agricultural expenditure in the national budget"
]



st.title("Google Policy Scraper — Multicountry & Custom Topics")
countries_input = st.text_input("Countries (separate multiple countries with commas):", "France, Italy")

topics_input = st.multiselect(    "Select or type topics:",    options=FAPDA_TOPICS,    default=[],    help="Start typing to filter topics.")
selected_topics = topics_input if topics_input else [t for t in default_topics if t in FAPDA_TOPICS]
start_date = st.text_input("Start Date (YYYY-MM-DD, optional):", "")
end_date = st.text_input("End Date (YYYY-MM-DD, optional):", "")
search_owner = st.text_input("User (optional):", "user")

st.caption("Searches will be conducted in English (Google 'hl=en')")
go_button = st.button("Scrape and Classify")
search_lang = "en"

@st.cache_resource
def load_policy_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_policy_model()

@st.cache_resource
def get_summary_pipeline():
    return pipeline("summarization", model="Falconsai/text_summarization", tokenizer="Falconsai/text_summarization", device=-1)

summary_pipe = get_summary_pipeline()

def is_url_excluded(url):
    domain = urlparse(url).netloc.lower()
    return any(excl in domain for excl in EXCLUDED_DOMAINS)

def extract_text_from_pdf(file, char_limit=PDF_TEXT_LIMIT):
    try:
        file.seek(0)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = " ".join([page.get_text("text") for page in doc])
        if not text.strip():
            file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(file.read())
                tmp_pdf_path = tmp_pdf.name
            try:
                result = subprocess.run(["pdftotext", tmp_pdf_path, "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                text = result.stdout.decode("utf-8")
            except Exception as e:
                st.warning(f"⚠️ pdftotext failed: {e}")
        return text[:char_limit] if text else None
    except Exception as e:
        st.warning(f"⚠️ Error reading PDF: {e}")
        return None

def clean_illegal_chars(text):
    return re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    label = "policy_decision" if probs[1] > probs[0] else "news"
    confidence = max(probs)
    return label, confidence

def refine_policy_decision(text):
    framework_indicators = [
        "framework", "reform", "strategy", "plan of action", "guideline",
        "policy structure", "national strategy", "policy framework",
        "regulatory framework", "institutional framework"
    ]
    decision_indicators = [
        "implemented", "introduced", "approved", "enacted", "adopted",
        "launched", "put into effect", "issued by", "effective from",
        "subsidy provided", "tax exemption granted"
    ]
    text_lower = text.lower()
    framework_score = sum(1 for kw in framework_indicators if kw in text_lower)
    decision_score = sum(1 for kw in decision_indicators if kw in text_lower)
    if framework_score >= 2 and framework_score > decision_score:
        return "policy_framework"
    return "policy_decision"

def drop_near_duplicates(df, text_col="text", url_col="Source 1 Link", text_threshold=0.85):
    df = df.copy()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    to_drop = set()
    for i in tqdm(range(len(df)), desc="🔍 Checking duplicates", unit="doc"):
        if i in to_drop:
            continue
        for j in range(i + 1, len(df)):
            if j in to_drop:
                continue
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
            if sim > text_threshold:
                to_drop.add(j)
    dropped_urls = df.iloc[list(to_drop)][url_col].tolist()
    print(f"🧹 Rimossi {len(to_drop)} duplicati approssimati.")
    print("URLs rimossi come duplicati:", dropped_urls)
    return df.drop(index=list(to_drop))

if go_button:
    countries = [c.strip() for c in countries_input.split(",") if c.strip()]
    topics = [t.strip() for t in topics_input if t.strip()] or FAPDA_TOPICS

    all_results = []
    excluded_documents = []
    st.info("🔍 Starting Google search and classification...")
    main_progress = st.progress(0)
    total_tasks = len(countries) * len(topics)
    task_idx = 0

    for country in countries:
        for topic in topics:
            task_idx += 1
            main_progress.progress(task_idx / total_tasks)
            date_filter = f" after:{start_date}" if start_date else ""
            date_filter += f" before:{end_date}" if end_date else ""
            query = f"{country} {topic} policy {date_filter}"
            st.write(f"**🔎 Query:** {query}")
            try:
                time.sleep(random.uniform(10, 20))
                urls = search(query, num_results=5, lang=search_lang, safe="off")
            except Exception as e:
                st.warning(f"⚠️ Google search error: {e}")
                continue

            for i, url in enumerate(urls):
                if is_url_excluded(url):
                    st.info(f"🚫 Skipped excluded domain: {url}")
                    continue

                with st.spinner(f"🔄 Processing {url}..."):
                    st.write(f"→ {url}")
                    try:
                        headers = {"User-Agent": "Mozilla/5.0"}
                        response = requests.get(url, headers=headers, timeout=10)
                        content_type = response.headers.get("Content-Type", "")

                        if url.endswith(".pdf") or "application/pdf" in content_type:
                            content = extract_text_from_pdf(io.BytesIO(response.content))
                        else:
                            soup = BeautifulSoup(response.text, "html.parser")
                            paragraphs = soup.find_all("p")
                            content = " ".join([p.get_text() for p in paragraphs])

                        if not content or len(content) < 100:
                            st.warning(f"⛔ Insufficient content from: {url}")
                            continue

                        try:
                            translated = GoogleTranslator(source="auto", target="en").translate(content[:4000])
                        except Exception as e:
                            st.warning(f"⚠️ Translation failed: {e}")
                            continue

                        translated = clean_illegal_chars(translated)
                        if country.lower() not in translated.lower():
                            st.warning(f"⛔ '{country}' not mentioned in document, skipped: {url}")
                            continue

                        if translated.lower().count(country.lower()) < 2:
                            st.warning(f"⛔ '{country}' mentioned too infrequently, skipped: {url}")
                            continue
                        label, confidence = classify_text(translated)
                        if label == "policy_decision":
                            st.write("🔁 Refining policy decision type...")
                            label = refine_policy_decision(translated)
                        st.write(f"🧪 Label: {label} (confidence={confidence:.2f})")
                        summary = ""
                        if confidence > 0.3:
                            try:
                                summary = summary_pipe(translated[:1024], max_new_tokens=130, min_length=30, do_sample=False)[0]["summary_text"]
                            except Exception as e:
                                st.warning(f"⚠️ Summarization failed: {e}")

                        all_results.append({
                            "User": search_owner,
                            "ID": "",
                            "Country Name": country,
                            "Initial date": "",
                            "End date": "",
                            "Policy 1": "",
                            "Policy 2": "",
                            "Policy direction": "",
                            "Policy phase": "",
                            "Policy decision details": "",
                            "Context or additional info": "",
                            "Policy decision making institution": "",
                            "Budget": "",
                            "Beneficiaries": "",
                            "Commodity 1": "",
                            "Commodity 2": "",
                            "Commodity 3": "",
                            "Additional Commodities": "",
                            "Term": "",
                            "Targeted": "",
                            "Emergency": "",
                            "Publish": "",
                            "Source 1 Name": "",
                            "Source 1 File": "",
                            "Source 1 Link": url,
                            "Publish Source 1": "",
                            "Source 2 Name": "",
                            "Source 2 File": "",
                            "Source 2 Link": "",
                            "Publish Source 2": "",
                            "Source 3 Name": "",
                            "Source 3 File": "",
                            "Source 3 Link": "",
                            "Publish Source 3": "",
                            "Giews Measure": "",
                            "topic": topic,
                            "query": query,
                            "publication_date": "",
                            "text": translated,
                            "label": label,
                            "confidence": round(confidence, 3),
                            "summary": summary
                        })
                    except Exception as e:
                        st.warning(f"❌ Error processing URL {url}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        df = drop_near_duplicates(df)
        output = io.BytesIO()
        df.to_excel(output, index=False, engine="openpyxl")
        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name="fapda_google_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No valid documents to export.")
