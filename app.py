"""AutoApply AI — Application Streamlit.

Projet PFA IGL4 :
- Saisie du user_id : charge automatiquement la dernière recherche si elle existe,
  ou ouvre directement le formulaire de nouvelle recherche s'il s'agit d'un nouvel utilisateur.
- Dépôt du CV PDF et des paramètres de recherche.
- Enregistrement dans MongoDB (GridFS, resumes, candidate_profiles, workflows).
- Extraction & construction du profil candidat.
- Recherche multi-sources parallélisée & filtrage par matching de pertinence.
- Affichage ordonné des offres d'emploi avec score, description et lien direct pour postuler manuellement.
- Aucune candidature automatique.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from backend import get_results, init_db, list_workflows, start_workflow

# ── Configuration de la page Streamlit ────────────────────────────────────────
st.set_page_config(
    page_title='AutoApply AI — Recommandation d’Emplois',
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Styles CSS Premium & Thème Moderne ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

/* Arrière-plan moderne dégradé nuit sombre */
.stApp {
    background: radial-gradient(circle at 20% 15%, #18192b 0%, #0d0e17 60%, #08090d 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(18, 20, 32, 0.85);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Cartes d'offres */
.job-card {
    background: rgba(25, 28, 45, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.job-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
}

/* Badges & Compétences */
.skill-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 4px 2px 0;
}
.skill-matched {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.skill-missing {
    background: rgba(148, 163, 184, 0.12);
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.25);
}
.source-badge {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.35);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
}

/* Boutons principaux */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Bouton lien d'offre */
.stLinkButton > a {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    text-align: center !important;
    border: none !important;
    padding: 10px 18px !important;
    display: inline-block !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
    transition: all 0.2s !important;
}
.stLinkButton > a:hover {
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Métriques */
[data-testid="stMetric"] {
    background: rgba(25, 28, 45, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: #f8fafc !important;
}

input, textarea, select {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Fonctions Utilitaires ──────────────────────────────────────────────────────
def _fmt_date(val) -> str:
    if isinstance(val, str):
        return val[:16].replace('T', ' ')
    if hasattr(val, 'strftime'):
        return val.strftime('%d/%m/%Y %H:%M')
    return 'Récent'


# ── Connexion MongoDB ──────────────────────────────────────────────────────────
try:
    init_db()
except Exception as exc:
    st.error(f'❌ Impossible de se connecter à MongoDB : {exc}')
    st.info('Vérifiez que le service MongoDB est bien actif sur localhost:27017.')
    st.stop()


# ── Barre Latérale (Sidebar) & Gestion de l'Utilisateur ────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <h2 style="margin:0; font-size: 1.6rem; font-weight:800; background: linear-gradient(135deg, #818cf8, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            💼 AutoApply AI
        </h2>
        <span style="font-size: 0.8rem; color: #94a3b8; font-weight:600;">PFA IGL4 • Recommandation de Jobs</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('### 👤 Identifiant Candidat')
    input_user_id = st.text_input(
        'User ID (Spring Boot / MySQL)',
        value=st.session_state.get('user_id', '42'),
        help='Référence externe de l’utilisateur issue du backend Spring Boot.',
    ).strip()

    # ── Logique de Changement d'Utilisateur ────────────────────────────────────
    last_user = st.session_state.get('last_user_id')
    user_changed = (input_user_id != last_user)

    if user_changed:
        st.session_state['last_user_id'] = input_user_id
        st.session_state['user_id'] = input_user_id

        # Vérifier si cet utilisateur a déjà des recherches enregistrées
        if input_user_id:
            try:
                user_history = list_workflows(input_user_id)
            except Exception:
                user_history = []
            
            if user_history:
                # S'il a un historique, charger automatiquement la dernière recherche
                st.session_state['workflow_id'] = user_history[0]['workflow_id']
            else:
                # S'il n'en a pas, réinitialiser pour ouvrir directement le formulaire
                st.session_state['workflow_id'] = None
        else:
            st.session_state['workflow_id'] = None

    user_id = st.session_state.get('user_id', '')

    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.1); border-left: 3px solid #6366f1; padding: 10px; border-radius: 4px; margin: 15px 0;">
        <p style="margin: 0; font-size: 0.78rem; color: #c7d2fe;">
            <strong>ℹ️ Mode Recommandation :</strong><br>
            Le système analyse votre CV, interroge 4 plateformes d'offres, calcule le score de pertinence et vous affiche les liens officiels pour postuler directement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Historique des Recherches MongoDB pour cet Utilisateur ────────────────
    st.markdown('### 📁 Historique des Recherches')
    workflows = []
    if user_id:
        try:
            workflows = list_workflows(user_id)
        except Exception as exc:
            st.caption(f'Erreur historique : {exc}')

        if workflows:
            wf_options = {
                f"{_fmt_date(w.get('created_at'))} • {w.get('status', 'ok').upper()}": w['workflow_id']
                for w in workflows
            }
            labels = list(wf_options.keys())
            current_wid = st.session_state.get('workflow_id')
            
            default_idx = 0
            for i, lb in enumerate(labels):
                if wf_options[lb] == current_wid:
                    default_idx = i
                    break

            chosen_label = st.selectbox('Recherches passées', labels, index=default_idx)
            
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                if st.button('📂 Charger', use_container_width=True):
                    st.session_state['workflow_id'] = wf_options[chosen_label]
                    st.rerun()
            with h_col2:
                if st.button('➕ Nouvelle', use_container_width=True):
                    st.session_state['workflow_id'] = None
                    st.rerun()
        else:
            st.info('Aucune recherche enregistrée pour cet utilisateur.')
            if st.button('🚀 Démarrer une recherche', use_container_width=True):
                st.session_state['workflow_id'] = None
                st.rerun()
    else:
        st.warning('Renseignez un User ID pour commencer.')

if not user_id:
    st.stop()


# ── En-tête Principal ──────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 25px; padding: 15px 0;">
    <h1 style="font-size: 2.4rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.5px;">
        🎯 Matching & Recommandation d'Emplois
    </h1>
    <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 0;">
        Analyse automatique de CV • Recherche multi-sources • Classement des offres par affinité
    </p>
</div>
""", unsafe_allow_html=True)


# ── SECTION 1 : Formulaire de Recherche & Dépôt du CV ─────────────────────────
# Déplié automatiquement si aucun résultat n'est actuellement affiché
is_form_expanded = (st.session_state.get('workflow_id') is None)

with st.expander('🚀 Lancer une nouvelle recherche personnalisée', expanded=is_form_expanded):
    with st.form('job_search_form', clear_on_submit=False):
        st.markdown('#### 1. Dépôt du CV (PDF)')
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            uploaded_file = st.file_uploader('Fichier CV au format PDF (max 8 Mo)', type=['pdf'])
        with f_col2:
            use_demo_cv = st.checkbox(
                '📄 Utiliser le CV démo (MyCv.pdf)',
                value=False,
                help='Permet de tester directement avec le CV exemple présent dans data/cvs/MyCv.pdf'
            )
            local_cv_path = st.text_input('Ou chemin absolu vers un PDF local', value='')

        st.divider()
        st.markdown('#### 2. Paramètres de Recherche & Filtrage')

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            target_roles_input = st.text_input(
                '🎯 Postes cibles recherchés (séparés par des virgules)',
                value='Python Developer, Full Stack Developer, Software Engineer',
                help='Exemple: Data Scientist, Backend Engineer, DevOps'
            )
            location_input = st.text_input(
                '📍 Ville ou Localisation souhaitée',
                value='Tunis',
                help='Ex: Tunis, Paris, Lyon, Remote'
            )
            country_code = st.text_input(
                '🌍 Code Pays (JSearch API)',
                value='tn',
                help='Code pays ISO à 2 lettres (tn = Tunisie, fr = France, us = USA...)'
            )

        with p_col2:
            employment_type = st.selectbox(
                '📋 Type de contrat',
                options=['', 'full_time', 'internship', 'part_time', 'contract'],
                format_func=lambda x: {
                    '': 'Tous types de contrats',
                    'full_time': 'CDI / Plein temps (Full-time)',
                    'internship': 'Stage / PFE (Internship)',
                    'part_time': 'Temps partiel (Part-time)',
                    'contract': 'Freelance / Contrat',
                }.get(x, x),
            )
            min_score_pref = st.slider(
                '🎯 Seuil de score minimal initial (%)',
                min_value=20,
                max_value=80,
                value=35,
                help='Les offres ayant un score inférieur ne seront pas pré-sélectionnées.'
            )
            max_per_src = st.slider(
                '🔢 Nombre max d’offres par source',
                min_value=3,
                max_value=20,
                value=5,
                help='Nombre d’annonces collectées par plateforme.'
            )

        st.divider()
        use_llm = st.toggle(
            '🧠 Matching sémantique par LLM (plus détaillé via OpenRouter)',
            value=False,
            help='Si désactivé, le système utilise l’algorithme heuristique instantané.'
        )

        submitted = st.form_submit_button('🔎 Analyser le CV et Trouver les Offres', use_container_width=True)

    if submitted:
        role_list = [r.strip() for r in target_roles_input.split(',') if r.strip()]
        if not role_list:
            st.error('Veuillez renseigner au moins un poste cible.')
            st.stop()

        cv_bytes: Optional[bytes] = None
        cv_name: str = 'cv.pdf'

        if uploaded_file is not None:
            cv_bytes = uploaded_file.getvalue()
            cv_name = uploaded_file.name
        elif use_demo_cv:
            demo_path = Path(__file__).resolve().parent / 'data' / 'cvs' / 'MyCv.pdf'
            if demo_path.is_file():
                cv_bytes = demo_path.read_bytes()
                cv_name = demo_path.name
            else:
                st.error(f'Fichier démo introuvable : {demo_path}')
                st.stop()
        elif local_cv_path.strip():
            p = Path(local_cv_path.strip())
            if p.is_file() and p.suffix.lower() == '.pdf':
                cv_bytes = p.read_bytes()
                cv_name = p.name
            else:
                st.error('Chemin local PDF invalide.')
                st.stop()
        else:
            st.error('Veuillez charger un fichier CV (PDF) ou cocher l’option CV démo.')
            st.stop()

        preferences = {
            'target_roles':           role_list,
            'employment_types':       [employment_type] if employment_type else [],
            'location':               location_input.strip() or None,
            'country':                country_code.strip().lower() or 'tn',
            'min_score':              min_score_pref,
            'max_results_per_source': max_per_src,
            'use_llm_matching':       use_llm,
        }

        with st.status('⏳ Exécution du pipeline...', expanded=True) as status:
            st.write('1️⃣ Sauvegarde du CV dans GridFS et MongoDB...')
            st.write('2️⃣ Construction et analyse du profil candidat...')
            st.write('3️⃣ Collecte des offres en parallèle (JSearch, Remotive, Arbeitnow, Greenhouse)...')
            st.write('4️⃣ Filtrage et calcul des scores de matching...')
            try:
                wid = start_workflow(user_id, cv_bytes, cv_name, preferences)
                st.session_state['workflow_id'] = wid
                status.update(label='✅ Analyse et recherche terminées avec succès !', state='complete')
                st.rerun()
            except Exception as exc:
                status.update(label=f'❌ Erreur : {exc}', state='error')
                st.error(f'Détail : {exc}')


# ── SECTION 2 : Affichage des Résultats ───────────────────────────────────────
active_wid = st.session_state.get('workflow_id')

if not active_wid:
    st.info('👈 Choisissez un CV ci-dessus et lancez une recherche, ou sélectionnez une recherche existante dans la barre latérale.')
    st.stop()

# Sécurité & Récupération du résultat
try:
    results_data = get_results(active_wid, user_id)
except Exception:
    # Le workflow en mémoire n'appartient pas au user_id actuel : on réinitialise proprement
    st.session_state['workflow_id'] = None
    st.rerun()

jobs = results_data.get('jobs', [])
profile = results_data.get('profile') or {}
found_count = results_data.get('found_count', 0)
top_score = max((j['match']['match_score'] for j in jobs), default=0)

st.divider()

# ── Bannière de rappel (Pas d'auto-apply) ──────────────────────────────────────
st.markdown("""
<div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 14px 18px; margin-bottom: 24px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 1.5rem;">📋</span>
        <div>
            <strong style="color: #34d399; font-size: 0.95rem;">Recommandation personnalisée — Aucune candidature automatique</strong><br>
            <span style="color: #cbd5e1; font-size: 0.85rem;">
                Toutes les offres ci-dessous sont triées par pertinence. Utilisez le bouton <strong>"Consulter l'offre & Postuler"</strong> pour accéder directement au formulaire de l'entreprise.
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Métriques Clés ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric('Offres Explorées', found_count)
m2.metric('Offres Qualifiées', len(jobs))
m3.metric('Meilleur Score', f'{top_score}%')
m4.metric('Statut Workflow', results_data.get('status', 'completed').capitalize())

# ── Profil Candidat Détecté ───────────────────────────────────────────────────
if profile:
    with st.expander('👤 Profil Candidat Détecté (Extrait du CV)', expanded=False):
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown(f"**Nom :** {profile.get('full_name') or 'Détecté dans le CV'}")
            st.markdown(f"**Email :** {profile.get('email') or 'Non spécifié'}")
            st.markdown(f"**Téléphone :** {profile.get('phone') or 'Non spécifié'}")
            st.markdown(f"**Localisation :** {profile.get('location') or 'Non spécifiée'}")
        with c_p2:
            if profile.get('target_roles'):
                st.markdown(f"**Rôles cibles :** {', '.join(profile['target_roles'])}")
            if profile.get('skills'):
                st.markdown('**Compétences identifiées :**')
                skills_html = ''.join(
                    f'<span class="skill-pill skill-matched">{s}</span>'
                    for s in profile['skills'][:25]
                )
                st.markdown(skills_html, unsafe_allow_html=True)

st.divider()

# ── SECTION 3 : Liste des Offres d'Emploi Ordonnées ───────────────────────────
st.markdown('### 💼 Offres Recommandées (Classées par Score)')

if not jobs:
    st.warning('Aucune offre ne correspond aux critères avec le seuil actuel. Essayez d’ajuster vos rôles cibles ou de baisser le seuil minimal.')
else:
    # ── Filtres interactifs rapides ───────────────────────────────────────────
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
    with filter_col1:
        search_query = st.text_input('🔍 Filtrer par mot-clé (titre, entreprise, compétence)', value='').strip().casefold()
    with filter_col2:
        score_filter = st.slider('Filtrer par Score Minimal (%)', 0, 100, 0)
    with filter_col3:
        all_sources = sorted({j['offer'].get('source', 'autre') for j in jobs})
        source_filter = st.selectbox('Source', ['Toutes'] + all_sources)

    # Application des filtres interactifs
    displayed_jobs = []
    for item in jobs:
        offer = item['offer']
        match = item['match']
        score = match.get('match_score', 0)

        if score < score_filter:
            continue
        if source_filter != 'Toutes' and offer.get('source') != source_filter:
            continue
        if search_query:
            searchable = f"{offer.get('title', '')} {offer.get('company', '')} {' '.join(match.get('matching_skills', []))}".casefold()
            if search_query not in searchable:
                continue

        displayed_jobs.append(item)

    st.caption(f'Affichage de **{len(displayed_jobs)}** offre(s) sur **{len(jobs)}**.')

    # ── Rendu des Cartes d'Offres ───────────────────────────────────────────────
    for idx, item in enumerate(displayed_jobs, start=1):
        offer = item['offer']
        match = item['match']
        score = match.get('match_score', 0)
        source = offer.get('source', 'web')

        # Détermination de la couleur selon le score
        if score >= 80:
            score_color = '#10b981'
            score_badge = f'🟢 {score}% MATCH'
        elif score >= 60:
            score_color = '#38bdf8'
            score_badge = f'🔵 {score}% MATCH'
        else:
            score_color = '#f59e0b'
            score_badge = f'🟡 {score}% MATCH'

        with st.container(border=True):
            # En-tête : Titre + Score
            head_col, score_col = st.columns([4, 1])
            with head_col:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                    <span style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 6px; font-weight:700; font-size: 0.85rem;">#{idx}</span>
                    <span class="source-badge">{source}</span>
                    <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700; color: #f8fafc;">
                        {offer.get('title', 'Sans titre')}
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                meta_parts = [f"🏢 **{offer.get('company', 'Entreprise')}**"]
                if offer.get('location'):
                    meta_parts.append(f"📍 {offer['location']}")
                if offer.get('job_type'):
                    meta_parts.append(f"💼 {offer['job_type']}")
                if offer.get('salary'):
                    meta_parts.append(f"💰 {offer['salary']}")
                st.markdown(' &nbsp;•&nbsp; '.join(meta_parts))

            with score_col:
                st.markdown(f"""
                <div style="text-align: right;">
                    <span style="font-size: 1.1rem; font-weight: 800; color: {score_color};">{score_badge}</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(score / 100.0)

            # Justification & Compétences
            if match.get('explanation'):
                st.markdown(f"💡 *{match['explanation']}*")

            m_skills = match.get('matching_skills', [])
            miss_skills = match.get('missing_skills', [])

            if m_skills or miss_skills:
                skills_html = '<div style="margin: 10px 0;">'
                for s in m_skills:
                    skills_html += f'<span class="skill-pill skill-matched">✓ {s}</span>'
                for s in miss_skills[:4]:
                    skills_html += f'<span class="skill-pill skill-missing">+ {s}</span>'
                skills_html += '</div>'
                st.markdown(skills_html, unsafe_allow_html=True)

            # Extrait de la description
            short_desc = item.get('short_desc', '').strip()
            if short_desc:
                st.markdown(f"<p style='color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;'>{short_desc}</p>", unsafe_allow_html=True)

            # Description complète (accordeon optionnel)
            full_desc = item.get('full_desc', '').strip()
            if full_desc and len(full_desc) > len(short_desc):
                with st.expander('📖 Lire la description complète de l’offre'):
                    st.write(full_desc)

            # ── Action : Lien direct pour postuler ─────────────────────────────
            action_col1, action_col2 = st.columns([1, 2])
            with action_col1:
                st.link_button(
                    "🔗 Consulter l'offre & Postuler",
                    url=str(offer.get('url', '#')),
                    use_container_width=True,
                )
            with action_col2:
                st.caption("Ce lien vous redirige directement sur l'annonce officielle de l'employeur.")


# ── SECTION 4 : Journal des alertes / erreurs techniques ──────────────────────
if results_data.get('errors'):
    with st.expander(f"⚠️ Avertissements techniques ({len(results_data['errors'])})"):
        for err in results_data['errors']:
            st.caption(f'• {err}')

st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 20px 0;">
    AutoApply AI — Projet PFA IGL4 • LangGraph + MongoDB + Streamlit • Recommandation d'offres manuelle
</div>
""", unsafe_allow_html=True)
