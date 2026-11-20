"""AutoApply AI — LangGraph + MongoDB + GridFS Backend.

Projet PFA IGL4 :
1. Extraction & construction du profil candidat depuis le CV (PDF -> GridFS + MongoDB).
2. Recherche d'offres multi-sources rapide et parallélisée (JSearch, Remotive, Arbeitnow, Greenhouse, fallback local).
3. Filtrage & matching intelligent (scoring heuristique et/ou LLM).
4. Recommandation et affichage ordonné des offres avec description, score et lien direct.
Aucune candidature automatique n'est soumise.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Optional, TypedDict
from urllib.parse import urlparse

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None
import gridfs
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pymongo import MongoClient

load_dotenv()

# ── Configuration & Connexions ────────────────────────────────────────────────
MONGO_URI       = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
DB_NAME         = os.getenv('MONGODB_DB') or os.getenv('MONGODB_DATABASE', 'autoapply_db')
MAX_CV_BYTES    = 8 * 1024 * 1024
DEFAULT_MIN_SCORE = int(os.getenv('MATCH_MIN_SCORE', '40'))
MAX_MATCH_JOBS  = int(os.getenv('MAX_MATCH_JOBS', '25'))

client       = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
db           = client[DB_NAME]
fs           = gridfs.GridFS(db)
checkpointer = MongoDBSaver(client, db_name=DB_NAME)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Initialisation de la Base de Données ──────────────────────────────────────
def init_db() -> None:
    """Vérifie la connexion MongoDB et s'assure des index nécessaires."""
    client.admin.command('ping')
    db.resumes.create_index([('user_id', 1), ('sha256', 1)], unique=True)
    db.candidate_profiles.create_index('user_id', unique=True)
    db.job_offers.create_index([('source', 1), ('source_id', 1)], unique=True)
    db.match_results.create_index([('user_id', 1), ('job_id', 1)], unique=True)
    db.workflows.create_index([('user_id', 1), ('created_at', -1)])


# ── Modèles Pydantic ──────────────────────────────────────────────────────────
class CandidateProfile(BaseModel):
    full_name:        Optional[str] = None
    email:            Optional[str] = None
    phone:            Optional[str] = None
    location:         Optional[str] = None
    skills:           list[str] = Field(default_factory=list)
    education:        list[str] = Field(default_factory=list)
    experience:       list[str] = Field(default_factory=list)
    languages:        list[str] = Field(default_factory=list)
    target_roles:     list[str] = Field(default_factory=list)
    employment_types: list[str] = Field(default_factory=list)


class JobOffer(BaseModel):
    source:           str
    source_id:        str
    title:            str
    company:          str
    location:         Optional[str] = None
    description:      Optional[str] = None
    tags:             list[str] = Field(default_factory=list)
    job_type:         Optional[str] = None
    category:         Optional[str] = None
    salary:           Optional[str] = None
    publication_date: Optional[str] = None
    url:              str


class MatchResult(BaseModel):
    job_id:          str
    match_score:     int = Field(ge=0, le=100)
    is_relevant:     bool
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills:  list[str] = Field(default_factory=list)
    explanation:     str = ''


# ── État LangGraph ────────────────────────────────────────────────────────────
class State(TypedDict, total=False):
    user_id:           str
    workflow_id:       str
    resume_id:         str
    cv_text:           str
    preferences:       dict[str, Any]
    candidate_profile: dict[str, Any]
    job_offers:        list[dict[str, Any]]
    ranked_jobs:       list[dict[str, Any]]
    errors:            list[str]


def config_for(workflow_id: str) -> dict:
    return {'configurable': {'thread_id': workflow_id}}


# ── Utilitaires & Nettoyage ───────────────────────────────────────────────────
def safe_url(url: str) -> bool:
    try:
        p = urlparse(str(url))
        return p.scheme in ('http', 'https') and bool(p.netloc) and not p.username
    except Exception:
        return False


def strip_html(value: Any) -> str:
    if not value:
        return ''
    cleaned = re.sub(r'<[^>]+>', ' ', str(value))
    return ' '.join(unescape(cleaned).split())[:15_000]


def extract_pdf_text(data: bytes) -> str:
    if fitz is None:
        raise RuntimeError(
            "Le module PyMuPDF est manquant dans l'environnement Python actif.\n"
            "Activez l'environnement dédié avec : conda activate autoapply-ai"
        )
    if not data.startswith(b'%PDF-'):
        raise ValueError('Le fichier ne semble pas être un PDF valide.')
    with fitz.open(stream=data, filetype='pdf') as doc:
        if doc.page_count > 35:
            raise ValueError('CV trop long (maximum 35 pages).')
        text = '\n'.join(page.get_text() for page in doc)
    if len(text.strip()) < 30:
        raise ValueError('PDF vide ou illisible. Si c’est un scan, un OCR est requis.')
    return text[:60_000]


# ── Stockage MongoDB du CV ───────────────────────────────────────────────────
def store_resume(user_id: str, filename: str, content: bytes) -> tuple[str, str]:
    """Enregistre le CV binaire dans GridFS et métadonnées dans MongoDB (collection resumes)."""
    if not content or len(content) > MAX_CV_BYTES:
        raise ValueError('Le CV doit être un fichier PDF non vide de 8 Mo maximum.')
    
    text = extract_pdf_text(content)
    digest = hashlib.sha256(content).hexdigest()
    
    prev = db.resumes.find_one({'user_id': user_id, 'sha256': digest})
    if prev:
        if not prev.get('extracted_text'):
            db.resumes.update_one({'_id': prev['_id']}, {'$set': {'extracted_text': text}})
        return str(prev['_id']), text
    
    gfs_id = fs.put(content, filename=filename, content_type='application/pdf', user_id=user_id)
    try:
        res = db.resumes.insert_one({
            'user_id': user_id,
            'gridfs_file_id': gfs_id,
            'filename': filename,
            'sha256': digest,
            'file_size': len(content),
            'extracted_text': text,
            'created_at': utcnow(),
        })
        return str(res.inserted_id), text
    except Exception:
        fs.delete(gfs_id)
        prev = db.resumes.find_one({'user_id': user_id, 'sha256': digest})
        if prev:
            return str(prev['_id']), text
        raise


# ── Intégration LLM (OpenRouter) ──────────────────────────────────────────────
def llm_keys() -> list[str]:
    return [k for i in (1, 2, 3) if (k := os.getenv(f'OPENROUTER_API_KEY_{i}', '').strip())]


def llm(index: int = 0) -> ChatOpenAI:
    keys = llm_keys()
    if not keys:
        raise RuntimeError('Ajoutez au moins une clé OPENROUTER_API_KEY_1 dans le fichier .env')
    model_name = os.getenv('OPENROUTER_MODEL', 'nvidia/nemotron-3.5-lightning:free')
    return ChatOpenAI(
        model=model_name,
        api_key=keys[index % len(keys)],
        base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
        temperature=0,
        timeout=18,
        max_retries=0,
    )


def structured_invoke(schema: type[BaseModel], prompt: str, first: int = 0):
    keys = llm_keys()
    if not keys:
        raise RuntimeError('Aucune clé OpenRouter configurée.')
    errors: list[str] = []
    # Teste jusqu'à 2 clés avec timeout court pour éviter de bloquer l'interface
    for step in range(min(2, len(keys))):
        i = (first + step) % len(keys)
        try:
            value = llm(i).with_structured_output(schema, method='function_calling').invoke(prompt)
            return schema.model_validate(value)
        except Exception as exc:
            errors.append(f'clé {i + 1}: {type(exc).__name__} ({str(exc)[:80]})')
    raise RuntimeError('OpenRouter indisponible : ' + '; '.join(errors))


# ── Extraction heuristique de secours du profil ──────────────────────────────
COMMON_TECH_SKILLS = [
    'Python', 'Java', 'C++', 'C#', 'JavaScript', 'TypeScript', 'PHP', 'Ruby', 'Go', 'Rust', 'Swift', 'Kotlin',
    'React', 'Angular', 'Vue', 'Next.js', 'Node.js', 'Express', 'Django', 'Flask', 'FastAPI', 'Spring Boot',
    'HTML', 'CSS', 'Tailwind', 'Bootstrap', 'Sass',
    'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Oracle', 'SQLite',
    'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Linux', 'Git', 'GitHub', 'CI/CD', 'Jenkins',
    'Machine Learning', 'Deep Learning', 'PyTorch', 'TensorFlow', 'Scikit-learn', 'NLP', 'Pandas', 'NumPy',
    'REST API', 'GraphQL', 'Microservices', 'Kafka', 'RabbitMQ', 'Agile', 'Scrum'
]


def heuristic_profile_extraction(text: str) -> CandidateProfile:
    """Extrait les informations essentielles du CV sans LLM en cas de panne réseau/quota."""
    profile = CandidateProfile()
    
    # Email
    email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    if email_match:
        profile.email = email_match.group(0)
    
    # Phone
    phone_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text)
    if phone_match:
        profile.phone = phone_match.group(0).strip()
    
    # Nom potentiel
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line) < 40 and not any(char in first_line for char in '@:/\\'):
            profile.full_name = first_line
    
    # Compétences
    text_lower = text.casefold()
    found_skills: list[str] = []
    for skill in COMMON_TECH_SKILLS:
        pattern = r'\b' + re.escape(skill.casefold()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    profile.skills = found_skills
    
    return profile


# ── NOEUD 1 : Construction du Profil Candidat ────────────────────────────────
def generate_candidate_profile(state: State) -> dict:
    """Analyse le CV avec LLM (avec fallback heuristique instantané) et fusionne les préférences."""
    errors = list(state.get('errors', []))
    cv_text = state.get('cv_text', '')
    profile: Optional[CandidateProfile] = None

    try:
        profile = structured_invoke(CandidateProfile, (
            'Extrais un profil professionnel structuré à partir du CV ci-dessous.\n'
            'Ne fabrique AUCUNE information. Si un champ est absent, laisse null ou liste vide.\n\n'
            f'CV :\n{cv_text[:20_000]}'
        ), first=0)
        if not isinstance(profile, CandidateProfile):
            profile = CandidateProfile.model_validate(profile)
    except Exception as exc:
        errors.append(f'LLM ({type(exc).__name__}), fallback extraction rapide activé.')
        profile = heuristic_profile_extraction(cv_text)

    # Fusion avec les préférences renseignées par l'utilisateur
    prefs = state.get('preferences', {})
    if prefs.get('target_roles'):
        profile.target_roles = list(prefs['target_roles'])
    if prefs.get('employment_types'):
        profile.employment_types = list(prefs['employment_types'])
    if prefs.get('location'):
        profile.location = str(prefs['location'])

    # Si aucune compétence trouvée, enrichir avec l'heuristique
    if not profile.skills:
        fallback = heuristic_profile_extraction(cv_text)
        profile.skills = fallback.skills

    return {'candidate_profile': profile.model_dump(), 'errors': errors}


# ── NOEUD 2 : Sauvegarde du Profil dans MongoDB ──────────────────────────────
def save_candidate_profile(state: State) -> dict:
    """Sauvegarde le profil candidat et les paramètres de recherche dans MongoDB."""
    db.candidate_profiles.update_one(
        {'user_id': state['user_id']},
        {
            '$set': {
                'profile': state['candidate_profile'],
                'preferences': state.get('preferences', {}),
                'resume_id': state.get('resume_id'),
                'updated_at': utcnow(),
            },
            '$setOnInsert': {'created_at': utcnow()},
        },
        upsert=True,
    )
    return {}


# ── Adaptateurs de Recherche d'Offres ─────────────────────────────────────────
def offer_from(raw: dict, source: str, **fields: Any) -> Optional[dict]:
    try:
        url = str(fields.get('url') or '')
        if not safe_url(url):
            return None
        data = JobOffer(
            source=source,
            source_id=str(fields['source_id']),
            title=str(fields.get('title') or 'Poste sans titre'),
            company=str(fields.get('company') or 'Entreprise confidentielle'),
            location=fields.get('location'),
            description=strip_html(fields.get('description')),
            tags=[str(v) for v in (fields.get('tags') or []) if v],
            job_type=fields.get('job_type'),
            category=fields.get('category'),
            salary=fields.get('salary'),
            publication_date=fields.get('publication_date'),
            url=url,
        )
        return data.model_dump(mode='json')
    except Exception:
        return None


def search_jsearch(roles: list[str], prefs: dict) -> list[dict]:
    key = os.getenv('RAPIDAPI_KEY')
    if not key:
        return []
    country  = str(prefs.get('country') or 'tn')
    location = str(prefs.get('location') or ('Tunisia' if country == 'tn' else country))
    result: list[dict] = []
    
    # Interroge le rôle principal rapidement
    for role in roles[:2]:
        try:
            resp = requests.get(
                'https://jsearch.p.rapidapi.com/search-v2',
                headers={
                    'x-rapidapi-key': key,
                    'x-rapidapi-host': os.getenv('RAPIDAPI_HOST', 'jsearch.p.rapidapi.com')
                },
                params={'query': f'{role} in {location}', 'country': country, 'num_pages': 1},
                timeout=8,
            )
            resp.raise_for_status()
            data_field = resp.json().get('data') or []
            raw_jobs = data_field.get('jobs') if isinstance(data_field, dict) else (data_field if isinstance(data_field, list) else [])
            for job in raw_jobs or []:
                offer = offer_from(
                    job, 'jsearch',
                    source_id=job.get('job_id') or job.get('job_apply_link'),
                    title=job.get('job_title'),
                    company=job.get('employer_name'),
                    location=job.get('job_location') or job.get('job_city') or job.get('job_country'),
                    description=job.get('job_description'),
                    tags=['remote'] if job.get('job_is_remote') else [],
                    job_type=job.get('job_employment_type'),
                    publication_date=job.get('job_posted_at_datetime_utc'),
                    url=job.get('job_apply_link'),
                )
                if offer:
                    result.append(offer)
        except Exception:
            pass
    return result


def search_remotive(roles: list[str]) -> list[dict]:
    result: list[dict] = []
    for role in roles[:2]:
        try:
            resp = requests.get('https://remotive.com/api/remote-jobs', params={'search': role}, timeout=8)
            resp.raise_for_status()
            for job in resp.json().get('jobs', []):
                offer = offer_from(
                    job, 'remotive',
                    source_id=job.get('id'),
                    title=job.get('title'),
                    company=job.get('company_name'),
                    location=job.get('candidate_required_location'),
                    description=job.get('description'),
                    tags=job.get('tags'),
                    job_type=job.get('job_type'),
                    category=job.get('category'),
                    salary=job.get('salary'),
                    publication_date=job.get('publication_date'),
                    url=job.get('url'),
                )
                if offer:
                    result.append(offer)
        except Exception:
            pass
    return result


def search_arbeitnow() -> list[dict]:
    result: list[dict] = []
    try:
        resp = requests.get('https://www.arbeitnow.com/api/job-board-api', timeout=8)
        resp.raise_for_status()
        for job in resp.json().get('data', []):
            created = job.get('created_at')
            date = datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if isinstance(created, (int, float)) else None
            offer = offer_from(
                job, 'arbeitnow',
                source_id=job.get('slug'),
                title=job.get('title'),
                company=job.get('company_name'),
                location=job.get('location'),
                description=job.get('description'),
                tags=job.get('tags'),
                job_type=', '.join(job.get('job_types') or []),
                publication_date=date,
                url=job.get('url'),
            )
            if offer:
                result.append(offer)
    except Exception:
        pass
    return result


def search_greenhouse(roles: list[str]) -> list[dict]:
    boards = [b.strip() for b in os.getenv('GREENHOUSE_BOARDS', 'stripe').split(',') if b.strip()][:2]
    words  = {w.casefold() for r in roles for w in re.findall(r'[\w+#.]{3,}', r)}
    result: list[dict] = []
    for board in boards:
        try:
            resp = requests.get(f'https://boards-api.greenhouse.io/v1/boards/{board}/jobs', timeout=6)
            resp.raise_for_status()
            for job in resp.json().get('jobs', []):
                title = str(job.get('title') or '')
                if words and not any(w in title.casefold() for w in words):
                    continue
                offer = offer_from(
                    job, 'greenhouse',
                    source_id=job.get('id'),
                    title=title,
                    company=board.capitalize(),
                    location=(job.get('location') or {}).get('name'),
                    description=title,
                    tags=[d.get('name') for d in job.get('departments', []) if d.get('name')],
                    publication_date=job.get('updated_at'),
                    url=job.get('absolute_url'),
                )
                if offer:
                    result.append(offer)
        except Exception:
            pass
    return result


def fallback_local_jobs() -> list[dict]:
    """Charge des offres locales de secours depuis data/jobs_for_matching.json en cas de panne réseau."""
    local_path = Path(__file__).resolve().parent / 'data' / 'jobs_for_matching.json'
    if not local_path.is_file():
        return []
    try:
        data = json.loads(local_path.read_text(encoding='utf-8'))
        valid: list[dict] = []
        for j in data:
            offer = offer_from(j, j.get('source', 'archive'), **j)
            if offer:
                valid.append(offer)
        return valid
    except Exception:
        return []


# ── NOEUD 3 : Recherche & Enregistrement des Offres (PARALLÉLISÉ) ──────────────
def job_search_agent(state: State) -> dict:
    """Collecte les offres multi-sources en parallèle pour une vitesse maximale."""
    profile = CandidateProfile.model_validate(state['candidate_profile'])
    prefs   = state.get('preferences', {})
    roles   = profile.target_roles or prefs.get('target_roles', ['Software Engineer'])
    per_src = max(2, min(int(prefs.get('max_results_per_source', 10)), 30))
    errors  = list(state.get('errors', []))
    found: dict[str, dict] = {}

    tasks = {
        'jsearch':    lambda: search_jsearch(roles, prefs),
        'remotive':   lambda: search_remotive(roles),
        'arbeitnow':  search_arbeitnow,
        'greenhouse': lambda: search_greenhouse(roles),
    }

    # Exécution concurrente des 4 sources
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                items = future.result(timeout=10)
                for offer in items[:per_src]:
                    key = f'{offer["source"]}:{offer["source_id"]}'
                    found[key] = offer
                    db.job_offers.update_one(
                        {'source': offer['source'], 'source_id': offer['source_id']},
                        {'$set': {'offer': offer, 'updated_at': utcnow()}},
                        upsert=True,
                    )
            except Exception as exc:
                errors.append(f'Source {name}: {str(exc)[:100]}')

    # Enrichissement avec la base locale si peu d'annonces en ligne
    if len(found) < 5:
        for offer in fallback_local_jobs()[:per_src]:
            key = f'{offer["source"]}:{offer["source_id"]}'
            if key not in found:
                found[key] = offer
                db.job_offers.update_one(
                    {'source': offer['source'], 'source_id': offer['source_id']},
                    {'$set': {'offer': offer, 'updated_at': utcnow()}},
                    upsert=True,
                )

    return {'job_offers': list(found.values()), 'errors': errors}


# ── NOEUD 4 : Filtrage & Scoring de Pertinence ────────────────────────────────
def heuristic_match(profile: CandidateProfile, offer: dict, prefs: Optional[dict] = None) -> MatchResult:
    """Algorithme de scoring robuste et équitable."""
    job_text = f'{offer["title"]} {offer.get("description") or ""} {" ".join(offer.get("tags") or [])}'.casefold()
    job_title = offer['title'].casefold()
    
    # 1. Correspondance de rôle (max parmi tous les rôles cibles)
    roles = profile.target_roles or ['developer']
    best_role_score = 0.0
    for r in roles:
        r_words = [w.casefold() for w in re.findall(r'[\w+#.]{3,}', r)]
        if not r_words:
            continue
        title_matches = sum(1 for w in r_words if w in job_title)
        body_matches  = sum(1 for w in r_words if w in job_text)
        
        r_score = (0.7 * (title_matches / len(r_words))) + (0.3 * (body_matches / len(r_words)))
        if r_score > best_role_score:
            best_role_score = r_score

    # 2. Compétences
    skills = [s.strip() for s in profile.skills if s.strip()]
    matches: list[str] = []
    for s in skills:
        pattern = r'\b' + re.escape(s.casefold()) + r'\b'
        if re.search(pattern, job_text):
            matches.append(s)
            
    missing = [s for s in skills if s not in matches][:6]
    skill_ratio = len(matches) / max(1, min(len(skills), 6))

    # 3. Bonus
    bonus = 0
    if 'remote' in job_text or 'télétravail' in job_text:
        bonus += 5
    if prefs and prefs.get('location'):
        loc_pref = str(prefs['location']).casefold()
        if loc_pref and loc_pref in str(offer.get('location') or '').casefold():
            bonus += 5

    raw_score = (50.0 * best_role_score) + (45.0 * min(1.0, skill_ratio)) + bonus
    score = int(round(max(0, min(100, raw_score))))

    explanation_parts = []
    if best_role_score >= 0.7:
        explanation_parts.append('Excellente correspondance de titre')
    elif best_role_score >= 0.4:
        explanation_parts.append('Bonne adéquation du rôle')
    else:
        explanation_parts.append('Rôle partiellement aligné')

    if matches:
        explanation_parts.append(f'{len(matches)} compétence(s) en phase')
    else:
        explanation_parts.append('Profil transverse')

    return MatchResult(
        job_id=f'{offer["source"]}:{offer["source_id"]}',
        match_score=score,
        is_relevant=True,
        matching_skills=matches[:12],
        missing_skills=missing,
        explanation=' • '.join(explanation_parts),
    )


def llm_match(profile: CandidateProfile, offer: dict) -> MatchResult:
    prompt = (
        'Évalue l’adéquation de cette offre d’emploi pour le profil candidat ci-dessous.\n'
        'Attribue un score de pertinence entre 0 et 100.\n'
        'Retourne les compétences correspondantes (matching_skills), les compétences manquantes (missing_skills),\n'
        'is_relevant (true si score >= 40) et une courte phrase d’explication.\n\n'
        f'Profil : {profile.model_dump_json()}\n'
        f'Offre : Titre: {offer["title"]} | Entreprise: {offer["company"]} | '
        f'Tags: {offer.get("tags", [])} | Description: {offer.get("description", "")[:2500]}'
    )
    resp = structured_invoke(MatchResult, prompt, first=1)
    result = MatchResult.model_validate(resp)
    result.job_id = f'{offer["source"]}:{offer["source_id"]}'
    return result


def matching_agent(state: State) -> dict:
    """Classe toutes les offres par score de pertinence et enregistre les résultats dans match_results."""
    profile    = CandidateProfile.model_validate(state['candidate_profile'])
    prefs      = state.get('preferences', {})
    use_llm    = bool(prefs.get('use_llm_matching', False))
    offers     = state.get('job_offers', [])
    min_score  = int(prefs.get('min_score', DEFAULT_MIN_SCORE))
    errors     = list(state.get('errors', []))

    ranked: list[dict] = []
    
    for offer in offers[:max(1, MAX_MATCH_JOBS)]:
        try:
            if use_llm:
                match = llm_match(profile, offer)
            else:
                match = heuristic_match(profile, offer, prefs)
        except Exception as exc:
            errors.append(f'Matching ({offer["title"][:25]}): fallback heuristique ({type(exc).__name__})')
            match = heuristic_match(profile, offer, prefs)

        match.is_relevant = match.match_score >= min_score
        match_doc = match.model_dump()

        db.match_results.update_one(
            {'user_id': state['user_id'], 'job_id': match.job_id},
            {
                '$set': {
                    'workflow_id': state['workflow_id'],
                    'match': match_doc,
                    'updated_at': utcnow(),
                }
            },
            upsert=True,
        )

        ranked.append({'offer': offer, 'match': match_doc})

    ranked.sort(key=lambda x: x['match']['match_score'], reverse=True)
    return {'ranked_jobs': ranked, 'errors': errors}


# ── Graphe LangGraph ──────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(State)
    builder.add_node('candidate_profile', generate_candidate_profile)
    builder.add_node('save_profile', save_candidate_profile)
    builder.add_node('job_search', job_search_agent)
    builder.add_node('matching', matching_agent)

    builder.add_edge(START, 'candidate_profile')
    builder.add_edge('candidate_profile', 'save_profile')
    builder.add_edge('save_profile', 'job_search')
    builder.add_edge('job_search', 'matching')
    builder.add_edge('matching', END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


# ── API Publique ──────────────────────────────────────────────────────────────
def start_workflow(user_id: str, cv_bytes: bytes, filename: str, preferences: dict) -> str:
    """Déclenche le pipeline complet pour l'utilisateur."""
    user_id = str(user_id).strip()
    if not user_id or len(user_id) > 128:
        raise ValueError('L’identifiant user_id est obligatoire (maximum 128 caractères).')

    resume_id, cv_text = store_resume(user_id, filename, cv_bytes)
    workflow_id = str(uuid.uuid4())

    db.workflows.insert_one({
        'workflow_id': workflow_id,
        'user_id': user_id,
        'resume_id': resume_id,
        'preferences': preferences,
        'status': 'running',
        'created_at': utcnow(),
        'updated_at': utcnow(),
    })

    state: State = {
        'user_id': user_id,
        'workflow_id': workflow_id,
        'resume_id': resume_id,
        'cv_text': cv_text,
        'preferences': preferences,
        'job_offers': [],
        'ranked_jobs': [],
        'errors': [],
    }

    try:
        graph.invoke(state, config=config_for(workflow_id))
    except Exception as exc:
        db.workflows.update_one(
            {'workflow_id': workflow_id},
            {'$set': {'status': 'error', 'error_message': str(exc)[:300], 'updated_at': utcnow()}},
        )
        raise

    db.workflows.update_one(
        {'workflow_id': workflow_id},
        {'$set': {'status': 'completed', 'updated_at': utcnow()}},
    )
    return workflow_id


def get_results(workflow_id: str, user_id: str) -> dict:
    """Récupère l'état complet du workflow avec les offres ordonnées."""
    wf_record = db.workflows.find_one({'workflow_id': workflow_id, 'user_id': str(user_id)})
    if not wf_record:
        raise ValueError('Workflow introuvable pour cet utilisateur.')

    snapshot = graph.get_state(config_for(workflow_id))
    values = snapshot.values or {}

    jobs = []
    for item in values.get('ranked_jobs', []):
        offer, match = item['offer'], item['match']
        desc = offer.get('description') or ''
        jobs.append({
            'offer': offer,
            'match': match,
            'short_desc': desc[:320] + ('…' if len(desc) > 320 else ''),
            'full_desc': desc,
        })

    return {
        'workflow_id': workflow_id,
        'status': wf_record.get('status', 'completed'),
        'jobs': jobs,
        'errors': values.get('errors', []),
        'found_count': len(values.get('job_offers', [])),
        'profile': values.get('candidate_profile'),
        'preferences': wf_record.get('preferences', {}),
        'finished': not bool(snapshot.next),
    }


def list_workflows(user_id: str) -> list[dict]:
    """Retourne la liste des recherches récentes pour un utilisateur."""
    return list(db.workflows.find(
        {'user_id': str(user_id)}, {'_id': 0}
    ).sort('created_at', -1).limit(20))


def start_from_path(user_id: str, pdf_path: str, preferences: dict) -> str:
    path = Path(pdf_path)
    if path.suffix.lower() != '.pdf' or not path.is_file():
        raise ValueError(f'Fichier PDF introuvable : {pdf_path}')
    return start_workflow(user_id, path.read_bytes(), path.name, preferences)
