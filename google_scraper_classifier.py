# AGGIORNAMENTO: Scraping batch automatico con memoria di avanzamento (5 topics per paese)

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
import torch.nn.functional as F
from deep_translator import GoogleTranslator
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import warnings
from dateutil.parser import parse
import fitz
import tempfile
import subprocess
import random

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "mike-83/longformer_policy_classifier"
PDF_TEXT_LIMIT = 5000
EXCLUDED_DOMAINS = ["soundcloud.com", "youtube.com", "youtu.be", "spotify.com", "vimeo.com", "facebook.com", "instagram.com"]

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

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    label = "policy_decision" if probs[1] > probs[0] else "news"
    confidence = max(probs)
    return label, confidence

def clean_illegal_chars(text):
    return re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
def save_and_continue(country, batch_number, results):
    df = pd.DataFrame(results)
    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    filename = f"results_{country}_batch{batch_number}.xlsx"
    with open(filename, "wb") as f:
        f.write(output.getvalue())
    st.write(f"✅ Batch {batch_number} salvato come {filename}")
    st.session_state.results.extend(results)
    st.session_state.batch_memory["topic_index"] += 5
    if st.session_state.batch_memory["topic_index"] >= len(st.session_state.topics):
        st.session_state.batch_memory["country_index"] += 1
        st.session_state.batch_memory["topic_index"] = 0
    st.rerun()

st.title("🌍 Policy Scraper — Auto batch (5 topics per country)")

if "batch_memory" not in st.session_state:
    st.session_state.batch_memory = {"country_index": 0, "topic_index": 0}
if "results" not in st.session_state:
    st.session_state.results = []

countries_input = st.text_input("Countries (comma separated):", "France, Italy")
topics_input = st.multiselect("Topics:", options=["Food subsidy", "Unconditional cash transfer", "Tax on fuel and water", "Price control", "Import tariff", "Export ban", "Nutrition and health policy", "Production subsidies"], default=[])
start_date = st.text_input("Start Date (YYYY-MM-DD):", "")
end_date = st.text_input("End Date (YYYY-MM-DD):", "")
search_owner = st.text_input("User:", "user")

if "countries" not in st.session_state:
    st.session_state.countries = [c.strip() for c in countries_input.split(",") if c.strip()]
if "topics" not in st.session_state:
    st.session_state.topics = topics_input or ["Food subsidy", "Unconditional cash transfer", "Tax on fuel and water"]

ci = st.session_state.batch_memory["country_index"]
ti = st.session_state.batch_memory["topic_index"]

if ci < len(st.session_state.countries):
    country = st.session_state.countries[ci]
    topics = st.session_state.topics
    batch_topics = topics[ti:ti + 5]
    st.subheader(f"🌐 {country} — Topics {ti + 1} to {ti + len(batch_topics)}")
    results = []
    for topic in batch_topics:
        query = f"{country} {topic} policy"
        if start_date:
            query += f" after:{start_date}"
        if end_date:
            query += f" before:{end_date}"
        st.write(f"🔍 Query: {query}")
        try:
            urls = search(query, num_results=3, lang="en", safe="off")
            for url in urls:
                if is_url_excluded(url):
                    continue
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if "application/pdf" in response.headers.get("Content-Type", ""):
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                content = " ".join([p.get_text() for p in soup.find_all("p")])
                if len(content) < 100:
                    continue
                translated = GoogleTranslator(source="auto", target="en").translate(content[:4000])
                translated = clean_illegal_chars(translated)
                label, confidence = classify_text(translated)
                summary = ""
                if confidence > 0.3:
                    try:
                        summary = summary_pipe(translated[:512], max_new_tokens=100, min_length=30, do_sample=False)[0]["summary_text"]
                    except:
                        pass
                results.append({
                    "Country": country,
                    "Topic": topic,
                    "URL": url,
                    "Label": label,
                    "Confidence": round(confidence, 2),
                    "Summary": summary
                })
        except Exception as e:
            st.warning(f"❌ Error: {e}")

    if results:
        batch_num = ti // 5 + 1
        save_and_continue(country, batch_num, results)
else:
    st.success("🎉 All countries and topic batches processed.")
