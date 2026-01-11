"""
TounesBot - Assistant Intelligent Tunisien
==========================================
Interface web simple et intuitive pour tous les Tunisiens
Supporte: Français, Anglais, Arabe (+ dialecte tunisien)
"""

from pathlib import Path
from typing import Dict, List, Sequence
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from langdetect import LangDetectException, detect
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================

st.set_page_config(
    page_title="TounesBot - مساعدك الذكي",
    page_icon="🇹🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# STYLES CSS (INTERFACE SIMPLE ET CLAIRE)
# ==========================================

st.markdown("""
<style>
    /* ===== COULEURS TUNISIENNES ===== */
    :root {
        --tunisie-rouge: #e80020;
        --tunisie-blanc: #ffffff;
        --success: #28a745;
        --warning: #fd7e14;
        --info: #17a2b8;
    }
    
    /* ===== FOND ===== */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* ===== LOGO ET TITRE ===== */
    .header-container {
        background: linear-gradient(135deg, #e80020 0%, #c8102e 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(232, 0, 32, 0.3);
    }
    
    .logo-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 10px;
    }
    
    .logo {
        font-size: 4em;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .main-title {
        font-size: 3em;
        font-weight: 900;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        font-size: 1.3em;
        color: rgba(255,255,255,0.95);
        margin-top: 10px;
    }
    
    .subtitle-ar {
        font-size: 1.5em;
        font-weight: 600;
        color: white;
        margin-top: 5px;
    }
    
    /* ===== CARTES SIMPLES ===== */
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
        border-left: 5px solid #e80020;
    }
    
    .info-card h3 {
        color: #e80020;
        margin-top: 0;
        font-size: 1.4em;
    }
    
    .info-card p {
        color: #555;
        line-height: 1.8;
        font-size: 1.05em;
    }
    
    /* ===== MESSAGES DE CHAT SIMPLES ===== */
    .chat-container {
        display: flex;
        margin: 20px 0;
        gap: 15px;
        align-items: flex-start;
    }
    
    .chat-container.user {
        flex-direction: row-reverse;
    }
    
    .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8em;
        flex-shrink: 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }
    
    .avatar.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .avatar.bot {
        background: linear-gradient(135deg, #e80020 0%, #ff6b6b 100%);
    }
    
    .message-wrapper {
        max-width: 70%;
    }
    
    .message-wrapper.user {
        text-align: right;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 25px;
        border-radius: 25px 25px 5px 25px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        font-size: 1.1em;
        line-height: 1.7;
    }
    
    .bot-message {
        background: white;
        color: #333;
        padding: 25px 30px;
        border-radius: 25px 25px 25px 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        border-left: 5px solid #e80020;
        font-size: 1.25em;
        line-height: 1.8;
        font-weight: 500;
    }
    
    .timestamp {
        font-size: 0.8em;
        color: #999;
        margin: 8px 15px;
    }
    
    /* ===== BADGES DE CONFIANCE SIMPLES ===== */
    .confidence-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 1em;
        font-weight: 700;
        margin: 15px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #fd7e14 0%, #ffc107 100%);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
        color: white;
    }
    
    /* ===== SECTION D'EXEMPLES ===== */
    .example-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 20px;
        margin: 25px 0;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    .example-section h3 {
        margin: 0 0 15px 0;
        font-size: 1.5em;
        text-align: center;
    }
    
    .example-question {
        background: rgba(255,255,255,0.2);
        padding: 12px 18px;
        border-radius: 15px;
        margin: 10px 0;
        font-size: 1.05em;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-question:hover {
        background: rgba(255,255,255,0.3);
        transform: translateX(5px);
    }
    
    /* ===== ALTERNATIVES SIMPLES ===== */
    .alternative-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 4px solid #17a2b8;
        transition: all 0.3s ease;
    }
    
    .alternative-box:hover {
        background: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* ===== FOOTER SIMPLE ===== */
    .footer {
        text-align: center;
        padding: 40px 20px;
        color: #666;
        border-top: 3px solid #e80020;
        margin-top: 50px;
    }
    
    .footer-title {
        font-size: 1.3em;
        font-weight: 700;
        color: #e80020;
        margin-bottom: 10px;
    }
    
    /* ===== BOUTONS ===== */
    .stButton > button {
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.05em;
        padding: 12px 30px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2em;
        }
        
        .message-wrapper {
            max-width: 85%;
        }
        
        .bot-message {
            font-size: 1.1em;
        }
        
        .logo {
            font-size: 3em;
        }
    }
    
    /* ===== MASQUER LES ÉLÉMENTS TECHNIQUES ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def _resolve_existing_path(base_dir: Path, candidates: Sequence[str]) -> Path:
    """Retourne le premier chemin existant."""
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return base_dir / candidates[0]

@st.cache_data(show_spinner="⏳ Chargement...")
def load_data(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Charge les données et embeddings."""
    data_path = _resolve_existing_path(
        data_dir,
        ["tunisian_assistant_data_clean.csv", "tunisian_assistant_data.csv"],
    )
    embeddings_path = _resolve_existing_path(
        models_dir,
        ["question_embeddings_final.npy", "question_embeddings.npy"],
    )
    
    df = pd.read_csv(data_path)
    embeddings = np.load(embeddings_path)
    return df, embeddings

@st.cache_resource(show_spinner="⏳ Préparation de l'assistant...")
def load_model(model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> SentenceTransformer:
    """Charge le modèle."""
    return SentenceTransformer(model_name)

def get_confidence_info(confidence: float) -> tuple[str, str, str]:
    """Retourne classe CSS, emoji et texte selon la confiance."""
    if confidence > 0.75:
        return "confidence-high", "🟢", "Excellente"
    elif confidence > 0.5:
        return "confidence-medium", "🟡", "Bonne"
    else:
        return "confidence-low", "🔴", "Faible"

# ==========================================
# CLASSE CHATBOT
# ==========================================

class TunisianAssistantChatbot:
    """Assistant intelligent pour les Tunisiens."""
    
    def __init__(
        self,
        data_loader=load_data,
        model_loader=load_model,
        tunisian_keywords: Sequence[str] | None = None,
    ):
        """Initialisation."""
        self.df, self.embeddings = data_loader()
        self.model = model_loader()
        
        self.tunisian_keywords = set(
            tunisian_keywords or [
                "kifech", "kifesh", "chnowa", "chnoua", "chneya", "chnou",
                "win", "wين", "b9adech", "9adech", "9adechou", "9adeh",
                "barsha", "برشا", "nheb", "نحب", "nحب",
                "3andou", "عندو", "3andek", "عندك", "hakka", "هكّا",
                "mta3", "متاع", "eli", "إلي", "اللي",
                "nجي", "nروح", "nعمل", "tعمل", "يعمل"
            ]
        )
    
    def _contains_tunisian_keyword(self, text: str) -> bool:
        """Vérifie les mots tunisiens."""
        lower_text = text.lower()
        return any(keyword in lower_text for keyword in self.tunisian_keywords)
    
    def detect_language(self, text: str) -> str:
        """Détecte la langue."""
        lower_text = text.lower()
        
        if self._contains_tunisian_keyword(lower_text):
            return "tn"
        
        try:
            detected = detect(text)
        except LangDetectException:
            detected = "unknown"
        
        if detected == "ar" and self._contains_tunisian_keyword(lower_text):
            return "tn"
        
        return detected
    
    def find_best_match(self, query: str, top_k: int = 3) -> List[Dict]:
        """Trouve les meilleures réponses."""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': self.df.iloc[idx]['question'],
                'reponse': self.df.iloc[idx]['reponse'],
                'categorie': self.df.iloc[idx]['categorie'],
                'langue': self.df.iloc[idx]['langue'],
                'similarity': float(similarities[idx]),
                'dialecte_tunisien': self.df.iloc[idx].get('dialecte_tunisien', False)
            })
        
        return results
    
    def chat(self, query: str, threshold: float = 0.5) -> Dict:
        """Génère une réponse."""
        detected_lang = self.detect_language(query)
        matches = self.find_best_match(query, top_k=3)
        best_match = matches[0]
        
        if best_match['similarity'] >= threshold:
            return {
                'response': best_match['reponse'],
                'category': best_match['categorie'],
                'confidence': best_match['similarity'],
                'detected_language': detected_lang,
                'alternative_matches': matches[1:],
                'success': True
            }
        else:
            return {
                'response': self._fallback_response(detected_lang),
                'category': 'unknown',
                'confidence': best_match['similarity'],
                'detected_language': detected_lang,
                'alternative_matches': matches,
                'success': False
            }
    
    def _fallback_response(self, lang: str) -> str:
        """Réponse par défaut."""
        responses = {
            'fr': "Désolé, je n'ai pas bien compris votre question. Pouvez-vous la reformuler autrement ?",
            'en': "Sorry, I didn't quite understand your question. Could you rephrase it?",
            'ar': "عذراً، لم أفهم سؤالك جيداً. هل يمكنك إعادة صياغته؟",
            'tn': "سامحني، ما فهمتش سؤالك مليح. تنجم تعاود تقولو بطريقة أخرى؟"
        }
        return responses.get(lang, responses['fr'])

# ==========================================
# INITIALISATION
# ==========================================

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = TunisianAssistantChatbot()

if 'history' not in st.session_state:
    st.session_state.history = []

chatbot = st.session_state.chatbot

# ==========================================
# EN-TÊTE AVEC LOGO
# ==========================================

st.markdown("""
<div class="header-container">
    <div class="logo-title">
        <span class="logo">🇹🇳</span>
        <h1 class="main-title">TounesBot</h1>
    </div>
    <p class="subtitle">Votre Assistant Intelligent Tunisien</p>
    <p class="subtitle-ar">مساعدك الذكي للحياة اليومية في تونس</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# CARTE D'INFORMATION
# ==========================================

st.markdown("""
<div class="info-card">
    <h3>🤝 Comment puis-je vous aider ?</h3>
    <p>
        Je peux répondre à vos questions sur la vie quotidienne en Tunisie :
        <br>🏥 Santé • 🏛️ Administration • 🚇 Transport • 🎓 Éducation
        <br>💼 Emploi • 🗺️ Tourisme • 💰 Économie • et plus encore !
    </p>
    <p style="font-size: 0.95em; color: #777;">
        💬 Posez votre question en <strong>français, anglais, arabe ou dialecte tunisien</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# EXEMPLES DE QUESTIONS
# ==========================================

st.markdown("""
<div class="example-section">
    <h3>💡 Exemples de questions que vous pouvez poser :</h3>
    <div class="example-question">🏥 "Comment réserver un rendez-vous chez le médecin ?"</div>
    <div class="example-question">🇹🇳 "Kifech nخلص facture STEG ?"</div>
    <div class="example-question">🚇 "9adech ticket mta3 le métro ?"</div>
    <div class="example-question">🏛️ "Comment obtenir un passeport ?"</div>
    <div class="example-question">🇬🇧 "Where can I find apartments for rent in Tunis?"</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FORMULAIRE DE QUESTION
# ==========================================

st.markdown("### 💬 Posez votre question")

with st.form(key='question_form', clear_on_submit=True):
    user_input = st.text_area(
        "",
        height=120,
        placeholder="✍️ Tapez votre question ici... (en français, anglais, arabe ou tunisien)",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submit_button = st.form_submit_button("🚀 Envoyer ma question", type="primary", use_container_width=True)
    
    with col2:
        random_button = st.form_submit_button("🎲 Surprise", use_container_width=True)
    
    with col3:
        clear_button = st.form_submit_button("🗑️ Effacer", use_container_width=True)
    
    # Gestion des boutons
    if random_button:
        user_input = chatbot.df.sample(1)['question'].values[0]
        submit_button = True
    
    if clear_button:
        st.session_state.history = []
        st.rerun()

# Traitement de la question
if submit_button and user_input:
    with st.spinner("🤔 Je réfléchis à votre question..."):
        result = chatbot.chat(user_input, threshold=0.5)
        
        st.session_state.history.append({
            'timestamp': datetime.now().strftime("%H:%M"),
            'question': user_input,
            'result': result
        })

# ==========================================
# AFFICHAGE DES CONVERSATIONS
# ==========================================

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 💬 Conversation")
    
    for i, item in enumerate(reversed(st.session_state.history)):
        result = item['result']
        
        # Question utilisateur
        st.markdown(f"""
        <div class="chat-container user">
            <div class="message-wrapper user">
                <div class="user-message">
                    {item['question']}
                </div>
                <div class="timestamp">{item['timestamp']}</div>
            </div>
            <div class="avatar user">👤</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Réponse du bot
        st.markdown(f"""
        <div class="chat-container bot">
            <div class="avatar bot">🤖</div>
            <div class="message-wrapper bot">
                <div class="bot-message">
                    {result['response']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Badge de confiance
        confidence = result['confidence']
        confidence_class, emoji, text = get_confidence_info(confidence)
        confidence_pct = confidence * 100
        
        st.markdown(f"""
        <div style="text-align: center; margin: 15px 0;">
            <span class="confidence-badge {confidence_class}">
                {emoji} Confiance de la réponse : {text} ({confidence_pct:.0f}%)
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternatives si disponibles
        if result['alternative_matches'] and len(result['alternative_matches']) > 0:
            with st.expander("📚 Voir d'autres réponses possibles"):
                for j, alt in enumerate(result['alternative_matches'], 1):
                    st.markdown(f"""
                    <div class="alternative-box">
                        <strong>💡 Autre réponse {j}</strong>
                        <p style="margin-top: 10px;">{alt['reponse']}</p>
                        <p style="color: #666; font-size: 0.9em; margin-top: 8px;">
                            📁 Catégorie : {alt['categorie']} | 
                            Pertinence : {alt['similarity']*100:.0f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
else:
    st.info("👋 Bienvenue ! Posez votre première question ci-dessus pour commencer.")

# ==========================================
# GUIDE D'UTILISATION
# ==========================================

with st.expander("❓ Comment utiliser TounesBot ?"):
    st.markdown("""
    ### 📖 Guide d'utilisation
    
    **1️⃣ Posez votre question**
    - Écrivez votre question dans la zone de texte ci-dessus
    - Vous pouvez écrire en français, anglais, arabe ou dialecte tunisien
    
    **2️⃣ Obtenez une réponse**
    - TounesBot vous donnera une réponse claire et précise
    - Un indicateur de confiance vous indique la fiabilité de la réponse
    
    **3️⃣ Explorez les alternatives**
    - Cliquez sur "Voir d'autres réponses possibles" pour plus d'options
    
    **4️⃣ Exemples de sujets**
    - 🏥 Santé : rendez-vous, hôpitaux, assurance
    - 🏛️ Administration : passeport, CIN, factures
    - 🚇 Transport : métro, train, tarifs
    - 🎓 Éducation : universités, inscriptions
    - 💼 Emploi : recherche, salaires
    - 🗺️ Tourisme : sites, hôtels, culture
    
    **💡 Astuce** : Utilisez le bouton "🎲 Surprise" pour découvrir des questions intéressantes !
    """)

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    <p class="footer-title">🇹🇳 TounesBot – Assistant Intelligent en Tunisie</p>
    <p>Un assistant numérique pour faciliter la vie quotidienne en Tunisie</p>
    <p style="color: #999; font-size: 0.9em; margin-top: 15px;">
        Informations générales • Multilingue • © 2026
    </p>
</div>
""", unsafe_allow_html=True)