# Assistant Intelligent Tunisien NLP

Application Streamlit multilingue (FR / EN / AR / TN) propulsée par Sentence Transformers pour répondre aux questions fréquentes sur la Tunisie. Le code charge un jeu de données pré-embeddé et effectue une recherche par similarité cosinus.

## Structure rapide
- `app/app.py` : application Streamlit (UI, logique de chat, chargement des ressources).
- `data/` : jeu de données CSV (`tunisian_assistant_data_clean.csv` ou `tunisian_assistant_data.csv`).
- `models/` : embeddings numpy (`question_embeddings_final.npy` ou `question_embeddings.npy`).
- `notebooks/` : exploration/embeddings (sources de données/embeddings si besoin).

## Prérequis
- Python 3.10.x ou 3.11.x (recommandé pour torch/faiss).
- Pip ou conda (environnement isolé conseillé).

## Installation
```bash
conda create -n tn-assistant python=3.10 -y
conda activate tn-assistant
pip install -r requirements.txt
```

## Données et embeddings
- Place les CSV dans `data/` :
  - `tunisian_assistant_data_clean.csv` (prioritaire) ou `tunisian_assistant_data.csv`.
- Place les embeddings dans `models/` :
  - `question_embeddings_final.npy` (prioritaire) ou `question_embeddings.npy`.
- Si tu ne peux pas créer `models/`, l’app essaie aussi `notebooks/question_embeddings_final.npy` ou `notebooks/question_embeddings.npy`.

## Lancer l'application
```bash
streamlit run app/app.py
```
Ensuite ouvre http://localhost:8501.

## Fonctionnement
- `load_data()` et `load_model()` sont mis en cache (streamlit) et injectés dans le chatbot. Les données et le modèle ne sont chargés qu’une fois.
- Les embeddings pré-calculés sont utilisés pour la similarité cosinus (pas de recalcul d’embeddings des questions stockées).
- Détection simple du dialecte tunisien via mots-clés (ex. kifech, b9adech, 9adech…).

## Modèles utilisés
- **Sentence Transformer** : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`  
  - Pourquoi : support multilingue (FR/EN/AR) performant, taille raisonnable, et compatible avec la recherche sémantique.
- **Embeddings pré-calculés** : `question_embeddings_final.npy` (ou `question_embeddings.npy`)  
  - Pourquoi : éviter de recalculer les embeddings à chaque démarrage et garantir la cohérence avec le CSV (mêmes lignes/ordre).

## Variables importantes
- Chemins : dérivés de `app/app.py` (`DATA_DIR = project_root/data`, `MODELS_DIR = project_root/models`).
- Modèle : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- Seuil de confiance ajustable dans la sidebar.

## Dépannage courant
- **Fichier manquant** : assure-toi que le CSV est dans `data/` et le `.npy` dans `models/` (ou `notebooks/`).
- **Téléchargement modèle lent** : pré-télécharge le modèle :
  ```bash
  python - <<'PY'
  from huggingface_hub import snapshot_download
  snapshot_download("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
  PY
  ```
- **Conflit de dépendances** : reste sur Python 3.10/3.11 et les versions du `requirements.txt` (incluant `huggingface-hub==0.16.4`).

## Tests rapides
- Depuis le shell Python (env activé) :
  ```bash
  python - <<'PY'
  from app.app import TunisianAssistantChatbot
  cb = TunisianAssistantChatbot()
  res = cb.chat("kifech na7i passport?", threshold=0.4)
  print(res["detected_language"], res["category"], res["confidence"])
  PY
  ```
- Dans l’UI : poser une question en tunisien (“kifech…”, “9adech…”) → langue détectée `tn` et réponse avec confiance affichée.

## Bonnes pratiques
- Ne pas déplacer les caches Streamlit hors du projet (sinon rechargements plus lents).
- Garder les embeddings pré-calculés alignés avec le CSV (même ordre de lignes).
- Utiliser un environnement propre pour éviter les conflits torch/huggingface.


rebuild...



