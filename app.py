"""
Lab 1 — Tokenisation, Embeddings et appel d'un LLM  (version Streamlit)

L'étudiant ne voit aucun code : il saisit des textes, règle des paramètres,
observe les résultats et répond aux questions. Les fonctions « solutions »
du notebook enseignant sont exécutées en coulisses.

Lancement :  streamlit run app.py
"""
from __future__ import annotations

import hmac
import html
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ════════════════════════════════════════════════════════════════════════════
#  1. FONCTIONS « SOLUTIONS » (issues du notebook enseignant)
# ════════════════════════════════════════════════════════════════════════════

# ---- Partie 1 : tokenisation ------------------------------------------------
def decouper(texte, enc):
    """Liste des tokens (chaînes) de `texte`."""
    ids = enc.encode(texte)
    return [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]


def fertilite(texte, enc):
    """Nombre moyen de tokens par mot."""
    mots = texte.split()
    return len(enc.encode(texte)) / len(mots) if mots else float("nan")


def compter_paires(ids):
    compteur = {}
    for a, b in zip(ids, ids[1:]):
        compteur[(a, b)] = compteur.get((a, b), 0) + 1
    return compteur


def fusionner(ids, paire, nouvel_id):
    resultat, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == paire:
            resultat.append(nouvel_id)
            i += 2
        else:
            resultat.append(ids[i])
            i += 1
    return resultat


def entrainer_bpe(texte, nb_fusions, stats=None):
    """Apprend `nb_fusions` fusions BPE (niveau octets). `stats` (liste) reçoit (paire, effectif)."""
    ids = list(texte.encode("utf-8"))
    merges = {}
    for k in range(nb_fusions):
        paires = compter_paires(ids)
        if not paires:
            break
        meilleure = max(paires, key=paires.get)
        nouvel_id = 256 + k
        if stats is not None:
            stats.append((meilleure, paires[meilleure]))
        ids = fusionner(ids, meilleure, nouvel_id)
        merges[meilleure] = nouvel_id
    return merges


def construire_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), nid in merges.items():
        vocab[nid] = vocab[a] + vocab[b]
    return vocab


def decoder(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def encoder(texte, merges):
    ids = list(texte.encode("utf-8"))
    for paire, nouvel_id in merges.items():
        ids = fusionner(ids, paire, nouvel_id)
    return ids


# ---- Partie 2 : embeddings --------------------------------------------------
def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def id_token(mot, tok):
    ids = tok.encode(mot)
    if len(ids) != 1:
        raise ValueError(f"{mot!r} n'est pas un token unique : {ids}")
    return ids[0]


def plus_proches(mot, En, tok, k=5):
    i = id_token(mot, tok)
    scores = En @ En[i]
    meilleurs = np.argsort(-scores)[1:k + 1]
    return [(tok.decode([j]), float(scores[j])) for j in meilleurs]


def analogie(a, b, c, En, tok, k=5):
    ids = [id_token(m, tok) for m in (a, b, c)]
    v = En[ids[0]] - En[ids[1]] + En[ids[2]]
    v = v / np.linalg.norm(v)
    scores = En @ v
    candidats = [j for j in np.argsort(-scores)[: k + 5] if j not in ids]
    return [(tok.decode([j]), float(scores[j])) for j in candidats[:k]]


def matrice_similarite(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Xn @ Xn.T


def voisin_le_plus_proche(S):
    S = np.array(S, dtype=float)
    np.fill_diagonal(S, -np.inf)
    return S.argmax(axis=1)


def top_k(q_emb, X_docs, k=3):
    scores = X_docs @ q_emb
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


# ---- Partie 3 : LLM / RAG ---------------------------------------------------
CONSIGNE_RAG = (
    "Réponds à la question en utilisant UNIQUEMENT les extraits ci-dessous. "
    "Si l'information n'y figure pas, réponds exactement « Je ne sais pas »."
)


def construire_prompt(question, passages, consigne=CONSIGNE_RAG):
    contexte = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"{consigne}\n\nExtraits :\n{contexte}\n\nQuestion : {question}"


def cout_usd(nb_in, nb_out, prix_in_par_million, prix_out_par_million):
    return nb_in / 1e6 * prix_in_par_million + nb_out / 1e6 * prix_out_par_million


# ════════════════════════════════════════════════════════════════════════════
#  2. RESSOURCES LOURDES (chargées à la demande, mises en cache)
# ════════════════════════════════════════════════════════════════════════════
ENCODAGES = ["cl100k_base", "o200k_base", "p50k_base", "r50k_base"]


@st.cache_resource(show_spinner="Chargement du tokenizer…")
def get_enc(nom):
    import tiktoken
    return tiktoken.get_encoding(nom)


@st.cache_resource(show_spinner="Chargement de GPT-2 (la première fois : environ 500 Mo)…")
def charger_gpt2():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    modele = AutoModel.from_pretrained("gpt2")
    E = modele.get_input_embeddings().weight.detach().numpy().copy()
    del modele
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    return tok, En


@st.cache_resource(show_spinner="Chargement du modèle d'embeddings de phrases…")
def charger_modele_emb():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner="Calcul des embeddings…")
def embeddings_phrases(phrases: tuple):
    return charger_modele_emb().encode(list(phrases), normalize_embeddings=True)


@st.cache_data(show_spinner="Entraînement du mini-BPE…")
def entrainer_cache(corpus: str, nb_fusions: int):
    stats = []
    merges = entrainer_bpe(corpus, nb_fusions, stats)
    return merges, stats


# ---- Accès à l'API Groq ------------------------------------------------------
MODELE_DEFAUT = "openai/gpt-oss-120b"


class ErreurConfig(RuntimeError):
    pass


def lire_cle():
    cle = st.session_state.get("w_cle_api", "") or ""
    if not cle:
        try:
            cle = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            cle = ""
    if not cle:
        cle = os.environ.get("GROQ_API_KEY", "")
    return cle.strip().strip('"').strip("'")


def appeler(**kwargs):
    """Appel à l'API avec nouvelles tentatives en cas de limite de débit."""
    import groq
    from groq import Groq

    cle = lire_cle()
    if not cle:
        raise ErreurConfig("Aucune clé API : renseignez-la dans la page « 3.0 Connexion API ».")
    client = Groq(api_key=cle)
    effort = st.session_state.get("w_raisonnement", "low")
    if effort != "aucun":
        kwargs.setdefault("extra_body", {"reasoning_effort": effort})
    modele = (st.session_state.get("w_modele") or MODELE_DEFAUT).strip()
    for essai in range(4):
        try:
            return client.chat.completions.create(model=modele, **kwargs)
        except groq.RateLimitError:
            attente = 10 * (essai + 1)
            st.toast(f"⏳ Limite de débit atteinte ; nouvelle tentative dans {attente} s…")
            time.sleep(attente)
    raise RuntimeError("Limite de débit persistante : attendez une minute puis réessayez.")


def diagnostiquer(e):
    """Affiche le même diagnostic que la cellule 3.0 du notebook."""
    try:
        import groq
    except ImportError:
        st.error("Le paquet `groq` n'est pas installé.")
        return
    if isinstance(e, ErreurConfig):
        st.warning(str(e))
    elif isinstance(e, groq.AuthenticationError):
        st.error("❌ 401 : clé invalide, supprimée ou mal copiée. Créez-en une nouvelle sur console.groq.com/keys.")
    elif isinstance(e, groq.PermissionDeniedError):
        st.error(f"❌ 403 : accès refusé pour cette clé ou ce modèle : {e}")
    elif isinstance(e, groq.NotFoundError):
        st.error(f"❌ 404 : modèle introuvable, vérifiez le nom du modèle : {e}")
    elif isinstance(e, groq.BadRequestError):
        st.error(f"❌ 400 : requête refusée (modèle retiré de l'offre gratuite, ou paramètre non géré) : {e}")
    elif isinstance(e, groq.APIConnectionError):
        st.error(f"❌ Problème réseau : {e}")
    else:
        st.error(f"❌ {e}")


def demander(prompt, system=None, temperature=1.0, max_tokens=400):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    rep = appeler(messages=messages, temperature=temperature, max_completion_tokens=max_tokens)
    return (rep.choices[0].message.content or ""), rep.usage, rep.choices[0].finish_reason


def tokens_entree_api(texte):
    rep = appeler(messages=[{"role": "user", "content": texte}], max_completion_tokens=1)
    return rep.usage.prompt_tokens


# ════════════════════════════════════════════════════════════════════════════
#  3. OUTILS D'INTERFACE
# ════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
.tok{display:inline-block;padding:2px 6px;margin:2px 1px;border-radius:5px;
     font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre;
     border:1px solid rgba(128,128,128,.35);font-size:.95rem;}
.c0{background:rgba(66,133,244,.25)} .c1{background:rgba(52,168,83,.25)}
.c2{background:rgba(251,188,5,.30)}  .c3{background:rgba(234,67,53,.22)}
.c4{background:rgba(171,71,188,.25)}
</style>
"""


def afficher_tokens(tokens, espaces_visibles=True):
    """Affiche les tokens sous forme de pastilles colorées."""
    morceaux = []
    for i, t in enumerate(tokens):
        s = html.escape(t)
        if espaces_visibles:
            s = s.replace(" ", "·")
        s = s.replace("\n", "⏎").replace("$", "&#36;")
        morceaux.append(f'<span class="tok c{i % 5}">{s}</span>')
    st.markdown("".join(morceaux) or "*(vide)*", unsafe_allow_html=True)


# -- Widgets persistants : l'état est conservé quand on change de page --------
def champ_texte(label, cle, defaut, hauteur=100, **kw):
    st.session_state.setdefault(cle, defaut)
    return st.text_area(label, key=cle, height=hauteur, **kw)


def champ_ligne(label, cle, defaut, **kw):
    st.session_state.setdefault(cle, defaut)
    return st.text_input(label, key=cle, **kw)


def curseur(label, cle, mn, mx, defaut, **kw):
    mx = max(mx, mn + (kw.get("step") or 1))          # évite min == max
    st.session_state.setdefault(cle, defaut)
    st.session_state[cle] = min(max(st.session_state[cle], mn), mx)   # reste dans les bornes
    return st.slider(label, mn, mx, key=cle, **kw)


def liste(label, cle, options, defaut, **kw):
    st.session_state.setdefault(cle, defaut)
    return st.selectbox(label, options, key=cle, **kw)


def multi(label, cle, options, defaut, **kw):
    st.session_state.setdefault(cle, defaut)
    return st.multiselect(label, options, key=cle, **kw)


def nombre(label, cle, defaut, **kw):
    st.session_state.setdefault(cle, defaut)
    return st.number_input(label, key=cle, **kw)


def case(label, cle, defaut, **kw):
    st.session_state.setdefault(cle, defaut)
    return st.checkbox(label, key=cle, **kw)


def lignes(texte):
    """Lignes non vides, sans toucher aux espaces initiaux."""
    return [l for l in texte.split("\n") if l.strip()]


def parse_libelle(texte, sep="|"):
    """'Libellé | texte' -> [(libellé, texte)]"""
    out = []
    for l in lignes(texte):
        if sep in l:
            a, b = l.split(sep, 1)
            out.append((a.strip(), b.strip()))
        else:
            out.append((f"#{len(out) + 1}", l.strip()))
    return out


# ════════════════════════════════════════════════════════════════════════════
#  4. QUESTIONS & RÉPONSES ATTENDUES (visibles seulement en mode enseignant)
# ════════════════════════════════════════════════════════════════════════════
QUESTIONS = {
    "1.1": (
        "Où se placent les espaces dans les tokens ? Que constatez-vous sur le premier mot par rapport aux suivants ?",
        "L'espace est **attaché au début du token qui suit** (par exemple `\" je\"`), et non un token à part. "
        "Le premier mot n'a pas d'espace devant, donc son token est différent de celui du même mot au milieu d'une "
        "phrase. Les mots fréquents forment souvent un seul token ; les mots moins fréquents (certains mots français "
        "ici) peuvent être fragmentés en plusieurs morceaux.",
    ),
    "1.2": (
        "(a) Quelle langue est la plus « chère » en tokens par mot ? (b) Que change `o200k_base` par rapport à "
        "`cl100k_base` ? (c) Quelles conséquences pratiques pour un service qui traite de l'arabe ou du tunisien ?",
        "**(a)** L'arabe et l'arabizi ont en général une fertilité nettement plus élevée que l'anglais (le français est "
        "intermédiaire), car ces langues sont moins représentées dans les données d'apprentissage du tokenizer.\n\n"
        "**(b)** `o200k_base` a un vocabulaire environ deux fois plus grand ; il réduit généralement le nombre de "
        "tokens pour les langues non anglaises (à vérifier sur le tableau obtenu).\n\n"
        "**(c)** Plus de tokens = **coût plus élevé**, **latence plus élevée**, **moins de contenu dans la fenêtre "
        "de contexte**, et souvent une qualité un peu moindre.",
    ),
    "1.3": (
        "(a) Pourquoi un LLM peut-il se tromper en comptant les « r » dans *strawberry* ? (b) Pourquoi l'arithmétique "
        "sur de grands nombres est-elle difficile ? (c) `hello`, ` hello` et `Hello` sont-ils le même token ?",
        "**(a)** Le modèle ne voit pas des lettres mais un ou quelques tokens : l'orthographe interne des tokens n'est "
        "pas directement accessible, il doit l'inférer de ce qu'il a appris.\n\n"
        "**(b)** Les nombres sont découpés en blocs (1 à 3 chiffres pour `cl100k_base`) sans lien direct avec la "
        "valeur ou la position des chiffres, ce qui complique les retenues et l'alignement.\n\n"
        "**(c)** Non : trois tokens distincts, avec trois IDs et donc trois embeddings différents (casse et espace "
        "initial changent le token).",
    ),
    "1.4": (
        "(a) Pourquoi `tiktoken` produit-il beaucoup moins de tokens que votre mini-BPE ? (b) Pourquoi votre mini-BPE "
        "ne compresse-t-il pas du tout la phrase arabe ? Quel lien avec la question 1.2 ?",
        "**(a)** Le vocabulaire de `tiktoken` compte ~100 000 fusions apprises sur un corpus gigantesque (contre "
        "quelques dizaines sur un petit texte), et il utilise en plus un pré-découpage par expressions régulières "
        "(mots, chiffres, ponctuation).\n\n"
        "**(b)** Aucune fusion arabe n'a été apprise : le corpus d'entraînement est en français. Chaque lettre arabe "
        "occupe 2 octets en UTF-8, donc autant de tokens que d'octets, soit environ **2 tokens par caractère**. "
        "C'est le même phénomène (en plus extrême) que la fertilité élevée de l'arabe en 1.2 : **un tokenizer "
        "compresse bien ce qu'il a vu pendant son entraînement**.",
    ),
    "2.2": (
        "Les voisins sont-ils tous « sémantiquement proches » ? Quels autres types de proximité observez-vous "
        "(casse, pluriels, sous-mots) ?",
        "On trouve des mots liés par le sens, mais aussi beaucoup de **variantes de forme** : casse (`King`), "
        "pluriel/dérivés (`kings`), voire des fragments de sous-mots. Un embedding de token **statique** mélange donc "
        "proximité de sens et proximité de forme, et ne dépend pas du contexte : `bank` (rivière) et `bank` (argent) "
        "ont le même vecteur. Les embeddings contextuels et de phrases corrigent en partie ce point.\n\n"
        "*Note enseignant (analogies) : sur GPT-2 le résultat de `king − man + woman` est **incertain** ; c'est "
        "l'occasion de discuter honnêtement des limites de Word2Vec et de l'importance d'exclure les mots d'entrée.*",
    ),
    "2.4": (
        "La proximité est-elle dominée par la **langue** ou par le **sens** ? Qu'est-ce que cela implique pour une "
        "recherche d'information multilingue ?",
        "Elle est dominée par le **sens** : une phrase est plus proche de sa traduction que d'une phrase de même "
        "langue sur un autre sujet (les blocs « chat » et « bourse » ressortent sur la matrice, indépendamment de la "
        "langue). On peut donc **poser une question dans une langue et retrouver des documents dans une autre**, "
        "avec un seul index. À nuancer : la qualité varie selon les langues et les dialectes (l'arabe dialectal est "
        "moins bien couvert que l'arabe standard).",
    ),
    "2.5": (
        "Que constatez-vous pour les paires « excellent / mauvais » et « excellent / pas excellent » ? Quelle en est "
        "la conséquence pour un système de recherche ou de RAG ?",
        "Les phrases **contradictoires** (antonymes, négation) restent souvent **très similaires** (bien plus "
        "qu'avec une phrase sur un autre sujet). Les embeddings capturent surtout la **thématique** plutôt que la "
        "polarité logique. Conséquence : la recherche par embeddings retrouve des passages *sur le bon sujet*, sans "
        "garantir qu'ils affirment ce que l'on cherche. D'où l'intérêt d'un **re-ranking** (cross-encoder ou LLM) "
        "et d'une vérification par le LLM en aval.",
    ),
    "3.2": (
        "Quelle température choisir pour (a) extraire des informations d'un document, (b) faire du brainstorming ? "
        "`temperature=0` garantit-elle exactement la même réponse ?",
        "**(a)** Extraction, classification, RAG factuel : température **basse (0 ou proche)** pour la "
        "reproductibilité. **(b)** Brainstorming, écriture créative : température **plus élevée**. Même à 0, le "
        "résultat n'est **pas strictement garanti identique** (calculs numériques en parallèle, évolutions côté "
        "serveur) : forte réduction de la variabilité, pas déterminisme absolu.",
    ),
    "3.3": (
        "(a) Quelle différence entre `max_completion_tokens` et la **fenêtre de contexte** du modèle ? (b) Avec "
        "15 tokens, la réponse est tronquée, voire **vide** : pourquoi, avec un modèle de raisonnement ?",
        "**(a)** `max_completion_tokens` limite le nombre de tokens **générés** (génération coupée avec "
        "`finish_reason = \"length\"`). La **fenêtre de contexte** est la taille maximale de *entrée + sortie* "
        "traitable dans un même appel : propriété du modèle, alors que `max_completion_tokens` se règle à chaque "
        "appel.\n\n"
        "**(b)** Un modèle de raisonnement produit d'abord des tokens de « réflexion » qui **comptent dans le budget "
        "de sortie** ; avec 15 tokens, le budget peut être épuisé avant que la réponse visible ne commence. "
        "Pour ces modèles, ne pas fixer un plafond trop bas.",
    ),
    "3.4": (
        "(a) Quelle colonne de `tiktoken` se rapproche le plus du décompte de l'API ? L'écart `API − o200k` est-il à "
        "peu près **constant** d'une langue à l'autre ? (b) Que représente cet écart ? (c) Que faut-il utiliser pour "
        "estimer un coût ou un quota ?",
        "**(a)** Le modèle (gpt-oss) s'appuie sur un tokenizer de la famille **o200k** (variante « harmony »), donc la "
        "colonne `o200k` est la plus proche et l'écart `API − o200k` doit être **à peu près constant** (alors que "
        "l'écart avec `cl100k` varie selon la langue). *À confirmer sur les résultats ; sinon le modèle choisi "
        "utilise un autre tokenizer.*\n\n"
        "**(b)** Le **surcoût de formatage du chat** : balises de rôles, éventuel message système par défaut, etc.\n\n"
        "**(c)** Toujours le champ `usage` renvoyé par l'API (ou l'outil de comptage du fournisseur), jamais le "
        "tokenizer d'un autre fournisseur. Sur l'offre gratuite, les quotas sont exprimés en tokens.",
    ),
    "3.5": (
        "(a) Qu'a répondu le LLM à la question sur la bibliothèque **sans** contexte, et avec ? (b) Que s'est-il "
        "passé pour la question absente de la base ? (c) Que se passe-t-il si la recherche ramène de **mauvais** "
        "passages ? Comment ajuster `k` ?",
        "**(a)** Sans contexte, le LLM ne peut pas connaître ces horaires fictifs : il répond de façon générique, "
        "refuse, ou **invente** (hallucination). Avec le RAG, il répond à partir du passage (samedi de 9h à 13h).\n\n"
        "**(b)** Grâce à la consigne « Je ne sais pas », il doit reconnaître l'absence d'information au lieu "
        "d'inventer (un modèle peut parfois déroger à la consigne, d'où l'intérêt de tester des questions hors "
        "base).\n\n"
        "**(c)** Le LLM répond à partir de passages non pertinents (« garbage in, garbage out ») ou dit ne pas "
        "savoir alors que l'information existe. Un `k` trop petit risque de manquer le bon passage ; un `k` trop "
        "grand ajoute du bruit et des tokens (coût / quota). Pistes : re-ranking, meilleur découpage des documents, "
        "seuil de score minimal.",
    ),
    "bilan": (
        "Décrivez en 5 à 8 lignes le pipeline complet de votre mini-RAG. Indiquez **où intervient chaque "
        "tokenizer** et **où intervient chaque embedding**. Y a-t-il un seul tokenizer dans ce pipeline ?",
        "*question* → **tokenizer du modèle d'embedding** → **embedding** de la question → recherche des top-k dans "
        "la matrice des embeddings des documents (calculée au préalable avec le **même modèle**) → construction du "
        "prompt → **tokenizer du LLM** → LLM → tokens de sortie détokenisés → réponse.\n\n"
        "Il y a donc **au moins deux tokenizers différents** (modèle d'embedding et LLM). Les embeddings interviennent "
        "à deux niveaux : embedding de phrases pour la recherche (2.2) et, à l'intérieur du LLM, la table "
        "d'embeddings de tokens (2.1). Points valorisés : même modèle d'embedding pour documents et requêtes ; "
        "coût lié aux tokens du LLM (passages injectés + réponse).",
    ),
}


def est_enseignant():
    return bool(st.session_state.get("_enseignant", False))


def question(qid):
    enonce, reponse = QUESTIONS[qid]
    label = "Question finale" if qid == "bilan" else f"Question {qid}"
    st.divider()
    st.markdown(f"#### 📝 {label}")
    st.markdown(enonce)
    champ_texte("Votre réponse", f"w_q_{qid}", "", hauteur=150,
                label_visibility="collapsed", placeholder="Rédigez votre réponse ici…")
    if est_enseignant():
        with st.expander("✅ Réponse attendue (visible enseignant uniquement)"):
            st.markdown(reponse)


# ════════════════════════════════════════════════════════════════════════════
#  5. PAGES
# ════════════════════════════════════════════════════════════════════════════
PHRASES_LANGUES = (
    "Anglais | Artificial intelligence is transforming education.\n"
    "Français | L'intelligence artificielle transforme l'éducation.\n"
    "Arabe | الذكاء الاصطناعي يغير التعليم.\n"
    "Arabizi tunisien | Aslema, chnouwa a7welek ? Ena mrigel."
)

CORPUS_BPE = (
    "L'intelligence artificielle transforme l'éducation. "
    "L'intelligence artificielle aide les étudiants à apprendre. "
    "Les étudiants utilisent l'intelligence artificielle pour apprendre plus vite. "
    "Les modèles de langage apprennent à prédire le mot suivant. "
)

PHRASES_EMB = (
    "FR-chat | chat | Le chat dort sur le canapé.\n"
    "EN-chat | chat | The cat is sleeping on the sofa.\n"
    "AR-chat | chat | القط ينام على الأريكة.\n"
    "FR-bourse | bourse | La bourse de Tunis a clôturé en hausse.\n"
    "EN-bourse | bourse | The Tunis stock exchange closed higher.\n"
    "AR-bourse | bourse | أغلقت بورصة تونس على ارتفاع."
)

DOCS_DEFAUT = (
    "Session de rattrapage : l'inscription se fait sur la plateforme ScolNet avant le 15 janvier ; aucune inscription n'est acceptée après cette date.\n"
    "Bibliothèque : ouverte du lundi au vendredi de 8h à 19h et le samedi de 9h à 13h ; fermée les jours fériés.\n"
    "Attestation de scolarité : à retirer au bureau B12 (scolarité) sous 48 heures après la demande en ligne ; prévoir une pièce d'identité.\n"
    "Mot de passe : la réinitialisation se fait sur le portail helpdesk.exemple.tn ; un lien valable 30 minutes est envoyé sur l'adresse mail personnelle.\n"
    "Rapport de stage : à déposer en PDF sur la plateforme avant la soutenance, avec le formulaire de validation signé par le tuteur de l'entreprise.\n"
    "Bourses : le service des bourses reçoit les étudiants le mardi et le jeudi de 10h à 12h au bâtiment administratif.\n"
    "Emploi du temps : publié sur l'ENT deux semaines avant chaque semestre ; toute modification est notifiée par mail.\n"
    "Laboratoire d'informatique : accès par badge, du lundi au vendredi de 8h à 18h ; les stations GPU sont réservables via le calendrier partagé."
)

PAIRES_DEFAUT = (
    "Le film est excellent. || Le film est mauvais.\n"
    "Le film est excellent. || Le film n'est pas excellent.\n"
    "Le film est excellent. || Le film est génial.\n"
    "Le film est excellent. || La recette du couscous est simple."
)

TESTS_PIEGES = "1234567\n2024\n hello\nhello\nHello\nHELLO\nstrawberry\ndef f(x):\\n    return x"


# ---- Accueil -----------------------------------------------------------------
def page_accueil():
    st.title("🧠 Lab 1 — Tokenisation, Embeddings et appel d'un LLM")
    st.markdown(
        "Bienvenue ! Dans cet atelier, **vous n'écrivez pas de code** : vous saisissez des textes, réglez des "
        "paramètres, observez le comportement des modèles et répondez aux questions."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Durée", "1 h 30")
    c2.metric("Niveau", "2A")
    c3.metric("Points", "20 (+1 bonus)")

    st.subheader("🎯 Objectifs")
    st.markdown(
        "1. Comprendre comment un texte devient une suite d'entiers (**tokenisation BPE**) et pourquoi cela a un "
        "impact sur le coût, les langues et les limites des LLM.\n"
        "2. Comprendre qu'un **embedding** est un vecteur dont la géométrie reflète le sens ; mesurer une "
        "similarité et faire une recherche sémantique.\n"
        "3. **Appeler un LLM** par API, maîtriser ses paramètres de base et relier les trois briques dans un mini-RAG."
    )
    st.subheader("🗺️ Organisation")
    st.table(pd.DataFrame({
        "Partie": ["1", "2", "3"],
        "Contenu": ["Tokenisation (tiktoken + mini-BPE)", "Embeddings (tokens, phrases, recherche sémantique)",
                    "Appel d'un LLM + mini-RAG"],
        "Durée": ["30 min", "30 min", "30 min"],
        "Points": [7, 6, 7],
    }).set_index("Partie"))
    st.subheader("📋 Comment travailler")
    st.markdown(
        "- Suivez les pages dans l'ordre via le **menu de gauche**.\n"
        "- Sur chaque page : modifiez les champs, lancez, **observez**, puis répondez à la question en bas de page.\n"
        "- Vos réponses sont conservées pendant la séance tant que vous ne fermez pas l'onglet. "
        "**Téléchargez-les régulièrement** depuis la page « 📤 Rendu ».\n"
        "- Pour la partie 3, vous avez besoin de **votre propre clé API Groq** (page 3.0)."
    )


# ---- 1.1 ---------------------------------------------------------------------
def page_11():
    st.header("1.1 Découvrir un tokenizer réel : `tiktoken`")
    st.markdown(
        "Un LLM ne lit ni lettres ni mots : il lit des **tokens**, des identifiants entiers issus d'un vocabulaire "
        "fixe. Saisissez un texte et choisissez un tokenizer pour voir comment il est découpé."
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        texte = champ_texte("Texte à tokeniser", "w_t11", "Bonjour, je m'appelle Sami et j'étudie l'IA à Tunis.")
    with c2:
        nom = liste("Encodage", "w_e11", ENCODAGES, "cl100k_base")
        vis = case("Rendre les espaces visibles (·)", "w_v11", True)
    if not texte.strip():
        st.info("Saisissez un texte.")
        return
    enc = get_enc(nom)
    ids = enc.encode(texte)
    tokens = decouper(texte, enc)
    m1, m2, m3 = st.columns(3)
    m1.metric("Nb tokens", len(ids))
    m2.metric("Nb caractères", len(texte))
    m3.metric("Fertilité (tokens/mot)", f"{fertilite(texte, enc):.2f}")
    st.markdown("**Découpage :**")
    afficher_tokens(tokens, vis)
    with st.expander("IDs des tokens"):
        st.code(str(ids))
    st.caption("Un « � » signale un token qui ne contient qu'une partie des octets d'un caractère (typique des "
               "langues peu représentées ou des emojis).")
    if enc.decode(ids) == texte:
        st.success("Aller-retour sans perte : decode(encode(texte)) == texte ✅")
    question("1.1")


# ---- 1.2 ---------------------------------------------------------------------
def page_12():
    st.header("1.2 La tokenisation n'est pas neutre : l'effet de la langue")
    st.markdown("**Fertilité** = nombre moyen de tokens par mot (mots séparés par des espaces). "
                "Vous pouvez modifier, ajouter ou supprimer des phrases (format : `Langue | phrase`, une par ligne).")
    saisie = champ_texte("Phrases à comparer", "w_phrases_langues", PHRASES_LANGUES, hauteur=140)
    noms = multi("Encodages à comparer", "w_enc12", ENCODAGES, ["cl100k_base", "o200k_base"])
    paires = parse_libelle(saisie)
    if not paires or not noms:
        st.info("Ajoutez au moins une phrase et un encodage.")
        return
    lignes_tab = []
    for langue, phrase in paires:
        for n in noms:
            e = get_enc(n)
            lignes_tab.append({"Langue": langue, "Encodage": n, "Tokens": len(e.encode(phrase)),
                               "Caractères": len(phrase), "Fertilité": round(fertilite(phrase, e), 2)})
    df = pd.DataFrame(lignes_tab)
    st.dataframe(df, hide_index=True)
    st.markdown("**Fertilité par langue et par encodage**")
    st.bar_chart(df.pivot(index="Langue", columns="Encodage", values="Fertilité").loc[[p[0] for p in paires]])
    question("1.2")


# ---- 1.3 ---------------------------------------------------------------------
def page_13():
    st.header("1.3 Les pièges de la tokenisation")
    st.markdown("Une chaîne par ligne (les espaces initiaux sont conservés ; écrivez `\\n` pour un saut de ligne).")
    c1, c2 = st.columns([3, 1])
    with c1:
        saisie = champ_texte("Chaînes à tester", "w_t13", TESTS_PIEGES, hauteur=220)
    with c2:
        nom = liste("Encodage", "w_e13", ENCODAGES, "cl100k_base")
        vis = case("Espaces visibles (·)", "w_v13", True)
    enc = get_enc(nom)
    for s in lignes(saisie):
        s_reel = s.replace("\\n", "\n")
        toks = decouper(s_reel, enc)
        st.markdown(f"`{s!r}` → **{len(toks)}** token(s)")
        afficher_tokens(toks, vis)
    question("1.3")


# ---- 1.4 ---------------------------------------------------------------------
def page_14():
    st.header("1.4 Un mini-BPE entraîné sous vos yeux")
    st.markdown(
        "**Principe** : on part des **octets UTF-8** (ids 0 à 255), on compte les paires adjacentes, on fusionne la "
        "plus fréquente en un nouveau token (256, 257…), et on répète. Pour encoder un nouveau texte, on rejoue "
        "les fusions **dans l'ordre d'apprentissage**."
    )
    corpus = champ_texte("Corpus d'entraînement", "w_corpus14", CORPUS_BPE, hauteur=130)
    c1, c2 = st.columns([2, 1])
    with c1:
        n = curseur("Nombre de fusions", "w_n14", 0, 500, 40)
    with c2:
        vis = case("Espaces visibles (·)", "w_v14", True)
    if len(corpus) > 20000:
        st.warning("Corpus long : l'entraînement peut être lent.")
    if not corpus:
        st.info("Saisissez un corpus.")
        return

    merges, stats = entrainer_cache(corpus, n)
    vocab = construire_vocab(merges)
    n_oct = len(corpus.encode("utf-8"))
    ids_corpus = encoder(corpus, merges)
    assert decoder(ids_corpus, vocab) == corpus
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fusions apprises", len(merges))
    m2.metric("Octets du corpus", n_oct)
    m3.metric("Tokens après encodage", len(ids_corpus))
    m4.metric("Compression (octets/token)", f"{n_oct / len(ids_corpus):.2f}")

    with st.expander("📜 Table des fusions apprises (dans l'ordre)"):
        aff = lambda b: vocab[b].decode("utf-8", errors="replace").replace(" ", "·")
        st.dataframe(pd.DataFrame([
            {"n°": k + 1, "Paire fusionnée": f"{aff(a)} + {aff(b)}", "Nouveau token": aff(nid),
             "Id": nid, "Occurrences": cnt}
            for k, ((a, b), nid), (_, cnt) in zip(range(len(merges)), merges.items(), stats)
        ]), hide_index=True)

    st.subheader("Tester le mini-BPE sur un texte")
    test = champ_ligne("Texte à encoder", "w_test14", "L'intelligence artificielle aide les étudiants.")
    if test:
        ids = encoder(test, merges)
        assert decoder(ids, vocab) == test
        st.markdown("**Découpage mini-BPE :**")
        afficher_tokens([vocab[i].decode("utf-8", errors="replace") for i in ids], vis)

    st.subheader("Comparaison avec `tiktoken` (cl100k_base)")
    ar = "الذكاء الاصطناعي يغير التعليم."
    enc = get_enc("cl100k_base")
    tests = [("Corpus d'entraînement", corpus), ("Votre texte", test or ""), ("Phrase arabe", ar)]
    st.dataframe(pd.DataFrame([
        {"Texte": nom, "Caractères": len(t), "Octets UTF-8": len(t.encode("utf-8")),
         "Tokens mini-BPE": len(encoder(t, merges)), "Tokens tiktoken": len(enc.encode(t))}
        for nom, t in tests if t
    ]), hide_index=True)

    with st.expander("🎁 Bonus 1.5 — Courbe de compression"):
        max_n = curseur("Nombre maximal de fusions de la courbe", "w_max14", 10, 500, 120)
        if st.button("Tracer la courbe"):
            m_max, _ = entrainer_cache(corpus, max_n)
            items = list(m_max.items())
            valeurs = sorted({0, *range(5, max_n + 1, max(1, max_n // 24)), len(items)})
            taux = []
            for v in valeurs:
                sous = dict(items[:v])          # entraîner v fusions = garder les v premières
                taux.append(n_oct / len(encoder(corpus, sous)))
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(valeurs, taux, marker="o")
            ax.set_xlabel("Nombre de fusions")
            ax.set_ylabel("Octets / tokens")
            ax.set_title("Compression du mini-BPE")
            ax.grid(alpha=.3)
            st.pyplot(fig)
            plt.close(fig)
    question("1.4")


# ---- 2.1 ---------------------------------------------------------------------
def page_21():
    st.header("2.1 Les embeddings de tokens (statiques) dans GPT-2")
    st.markdown(
        "La première couche d'un LLM est une **table de correspondance** : une matrice `E` de taille "
        "(vocabulaire × dimension). L'embedding du token `i` est la ligne `i`. Le mot doit être **un seul token** ; "
        "l'**espace initial** compte (`\" king\"` ≠ `\"king\"`)."
    )
    tok, En = charger_gpt2()
    st.caption(f"Matrice d'embeddings : {En.shape[0]} tokens × {En.shape[1]} dimensions")

    st.subheader("Voisins les plus proches")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        mots = champ_ligne("Mots (séparés par des virgules)", "w_mots21", "king, Paris, computer")
    with c2:
        k = curseur("k voisins", "w_k21", 3, 20, 6)
    with c3:
        espace = case("Ajouter l'espace initial", "w_esp21", True)
    for m in [x.strip() for x in mots.split(",") if x.strip()]:
        mot = (" " + m) if espace else m
        try:
            res = plus_proches(mot, En, tok, k)
        except ValueError:
            st.error(f"{mot!r} n'est pas un token unique : {tok.tokenize(mot)}. Essayez un autre mot ou "
                     "modifiez l'option « espace initial ».")
            continue
        st.markdown(f"**{mot!r}**")
        st.dataframe(pd.DataFrame([{"Token": repr(t), "Cosinus": round(s, 3)} for t, s in res]),
                     hide_index=True)

    with st.expander("🎁 Bonus 2.3 — Analogies (résultat non garanti)", expanded=False):
        st.markdown("Cherche les tokens proches de `E[a] − E[b] + E[c]`, en excluant a, b et c.")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            a = champ_ligne("a", "w_a21", "king")
        with a2:
            b = champ_ligne("b", "w_b21", "man")
        with a3:
            c = champ_ligne("c", "w_c21", "woman")
        with a4:
            ka = curseur("k", "w_ka21", 3, 15, 5)
        if st.button("Calculer l'analogie"):
            try:
                res = analogie(" " + a.strip(), " " + b.strip(), " " + c.strip(), En, tok, ka)
                st.dataframe(pd.DataFrame([{"Token": repr(t), "Cosinus": round(s, 3)} for t, s in res]),
                             hide_index=True)
            except ValueError as e:
                st.error(str(e))
    question("2.2")


# ---- 2.2 ---------------------------------------------------------------------
def page_22():
    st.header("2.2 Les embeddings de phrases")
    st.markdown(
        "Un modèle **Sentence-Transformers multilingue** (dimension 384) produit **un vecteur par phrase**. "
        "Modifiez les phrases (format : `étiquette | sujet | phrase`, une par ligne) ; la couleur et le score de "
        "précision dépendent du **sujet**."
    )
    saisie = champ_texte("Phrases", "w_phrases22", PHRASES_EMB, hauteur=180)
    items = []
    for l in lignes(saisie):
        parts = [p.strip() for p in l.split("|", 2)]
        if len(parts) == 3:
            items.append(tuple(parts))
        else:
            st.warning(f"Ligne ignorée (format attendu `étiquette | sujet | phrase`) : {l[:60]}")
    if len(items) < 3:
        st.info("Saisissez au moins 3 phrases.")
        return
    etiquettes, sujets, phrases = zip(*items)
    X = embeddings_phrases(tuple(phrases))
    st.caption(f"Forme des embeddings : {X.shape}")

    S = matrice_similarite(X)
    vois = voisin_le_plus_proche(S)
    precision = np.mean([sujets[vois[i]] == sujets[i] for i in range(len(items))])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matrice de similarité (cosinus)**")
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(S, vmin=min(0, S.min()), vmax=1, cmap="viridis")
        ax.set_xticks(range(len(items)), etiquettes, rotation=45, ha="right")
        ax.set_yticks(range(len(items)), etiquettes)
        if len(items) <= 10:
            for i in range(len(items)):
                for j in range(len(items)):
                    ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if S[i, j] < .6 else "black")
        fig.colorbar(im, label="cosinus")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("**Projection PCA en 2D**")
        from sklearn.decomposition import PCA
        P = PCA(n_components=2).fit_transform(X)
        uniques = list(dict.fromkeys(sujets))
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        for u_i, u in enumerate(uniques):
            idx = [i for i, s in enumerate(sujets) if s == u]
            ax.scatter(P[idx, 0], P[idx, 1], s=80, color=cmap(u_i % 10), label=u)
        for (x, y), et in zip(P, etiquettes):
            ax.annotate(et, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.legend(title="Sujet")
        ax.grid(alpha=.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.metric("Le plus proche voisin a le même sujet dans", f"{precision * 100:.0f} % des cas")

    with st.expander("🔤 Le modèle d'embedding a SON PROPRE tokenizer (lien avec la partie 1)"):
        choix = st.selectbox("Phrase", phrases)
        afficher_tokens(charger_modele_emb().tokenizer.tokenize(choix), False)
    question("2.4")


# ---- 2.3 ---------------------------------------------------------------------
EXEMPLES_REQUETES = [
    "Quand ferme la bibliothèque le samedi ?",
    "J'ai oublié mon mot de passe",
    "How can I register for the retake exam?",
    "أين أجد جدول الحصص؟",
]


def docs_courants():
    return lignes(st.session_state.get("w_docs", DOCS_DEFAUT))


def page_23():
    st.header("2.3 Recherche sémantique sur un mini-corpus")
    st.markdown(
        "Base de connaissances **fictive** : aucune de ces informations n'existe dans les données d'un LLM. "
        "Vous pouvez la modifier (un document par ligne) : elle sera aussi utilisée par le mini-RAG (3.5)."
    )
    with st.expander("📚 Base de documents", expanded=False):
        champ_texte("Un document par ligne", "w_docs", DOCS_DEFAUT, hauteur=260)
    docs = docs_courants()
    if not docs:
        st.info("La base est vide.")
        return
    X_docs = embeddings_phrases(tuple(docs))
    st.caption(f"{len(docs)} documents · embeddings {X_docs.shape}")

    st.markdown("**Exemples de requêtes (y compris multilingues) :**")
    cols = st.columns(len(EXEMPLES_REQUETES))
    for col, q in zip(cols, EXEMPLES_REQUETES):
        col.button(q, key=f"ex23_{q}",
                   on_click=lambda q=q: st.session_state.update(w_req23=q))
    c1, c2 = st.columns([4, 1])
    with c1:
        requete = champ_ligne("Votre requête", "w_req23", EXEMPLES_REQUETES[0])
    with c2:
        k = curseur("k résultats", "w_k23", 1, len(docs), min(3, len(docs)))
    if requete.strip():
        q_emb = embeddings_phrases((requete,))[0]
        for rang, (i, s) in enumerate(top_k(q_emb, X_docs, k), 1):
            st.markdown(f"**#{rang} — cosinus {s:.2f}** · document {i + 1}")
            st.progress(float(max(0.0, min(1.0, s))))
            st.write(docs[i])


# ---- 2.5 ---------------------------------------------------------------------
def page_25():
    st.header("2.5 Les limites des embeddings")
    st.markdown("Comparez des paires de phrases (format : `phrase A || phrase B`, une paire par ligne).")
    saisie = champ_texte("Paires", "w_paires25", PAIRES_DEFAUT, hauteur=160)
    paires = [tuple(p.strip() for p in l.split("||", 1)) for l in lignes(saisie) if "||" in l]
    if not paires:
        st.info("Saisissez au moins une paire.")
        return
    rows = []
    for a, b in paires:
        ea, eb = embeddings_phrases((a, b))
        rows.append({"Cosinus": round(cosine(ea, eb), 3), "Phrase A": a, "Phrase B": b})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    st.bar_chart(df.assign(Paire=[f"{i + 1}" for i in range(len(df))]).set_index("Paire")["Cosinus"])
    question("2.5")


# ---- 3.0 ---------------------------------------------------------------------
def page_30():
    st.header("3.0 Connexion à l'API Groq")
    st.markdown(
        "Chaque étudiant utilise **sa propre clé** (offre gratuite, sans carte bancaire).\n\n"
        "1. Allez sur **console.groq.com** et connectez-vous.\n"
        "2. Menu **API Keys** → **Create API Key** → nommez-la (ex. `lab1`).\n"
        "3. **Copiez la clé immédiatement** (elle commence par `gsk_…`) et collez-la ci-dessous.\n\n"
        "🔒 La clé reste en mémoire dans votre session : elle n'est ni affichée, ni enregistrée, ni incluse dans "
        "votre rendu."
    )
    st.session_state.setdefault("w_cle_api", "")
    st.text_input("Clé API Groq", type="password", key="w_cle_api", placeholder="gsk_…")
    with st.expander("⚙️ Paramètres avancés (modèle)"):
        st.markdown("L'offre gratuite évolue : si vous obtenez une erreur « model not found / decommissioned », "
                    "choisissez un modèle listé sur console.groq.com/docs/models (ex. `openai/gpt-oss-20b`).")
        champ_ligne("Modèle", "w_modele", MODELE_DEFAUT)
        liste("Effort de raisonnement", "w_raisonnement", ["low", "medium", "high", "aucun"], "low",
              help="Les modèles « gpt-oss » réfléchissent avant de répondre. « aucun » : pour un modèle sans ce paramètre.")
    cle = lire_cle()
    if cle:
        st.caption(f"Clé lue : longueur {len(cle)}, format " + ("OK (gsk_…)" if cle.startswith("gsk_") else "inattendu"))
    if st.button("Tester la connexion", type="primary"):
        try:
            appeler(messages=[{"role": "user", "content": "ok"}], max_completion_tokens=50)
            st.success("✅ Connexion à l'API OK")
        except Exception as e:
            diagnostiquer(e)


# ---- 3.1 / 3.3 ---------------------------------------------------------------
def page_31():
    st.header("3.1 Bac à sable LLM : messages, paramètres, limite de sortie")
    st.markdown(
        "Avec une API « chat », on envoie une **liste de messages** (`system`, `user`) et on lit la réponse ainsi "
        "que le décompte de tokens (`usage`)."
    )
    system = champ_texte("Prompt système (optionnel)", "w_sys31", "", hauteur=80,
                         placeholder="Ex. : Tu es un professeur d'IA. Réponds en deux phrases maximum, en français, avec un exemple concret.")
    prompt = champ_texte("Prompt utilisateur", "w_p31", "Explique en deux phrases ce qu'est un token pour un LLM.", hauteur=90)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        T = curseur("Température", "w_T31", 0.0, 2.0, 1.0, step=0.1)
    with c2:
        mt = curseur("max_completion_tokens", "w_mt31", 1, 2000, 400)
    with c3:
        stream = case("Streaming", "w_st31", False)
    st.button("⚡ Tester avec max_completion_tokens = 15", on_click=lambda: st.session_state.update(w_mt31=15))

    if st.button("Envoyer", type="primary"):
        try:
            if stream:
                msgs = ([{"role": "system", "content": system}] if system.strip() else []) + \
                       [{"role": "user", "content": prompt}]
                flux = appeler(messages=msgs, temperature=T, max_completion_tokens=mt, stream=True)

                def gen():
                    for m in flux:
                        if m.choices and m.choices[0].delta.content:
                            yield m.choices[0].delta.content

                st.write_stream(gen())
                st.caption("Le décompte `usage` n'est pas affiché en mode streaming.")
            else:
                with st.spinner("Appel du LLM…"):
                    texte, usage, fin = demander(prompt, system.strip() or None, T, mt)
                if texte.strip():
                    st.markdown("**Réponse :**")
                    st.write(texte)
                else:
                    st.warning("Réponse **vide**.")
                a, b, c, d = st.columns(4)
                a.metric("Tokens en entrée", usage.prompt_tokens)
                b.metric("Tokens en sortie", usage.completion_tokens)
                c.metric("Total", usage.total_tokens)
                d.metric("finish_reason", fin)
                if fin == "length":
                    st.info("La génération a été coupée par la limite de sortie (`finish_reason = length`). "
                            "Avec un modèle de raisonnement, les tokens de « réflexion » comptent aussi dans ce budget.")
                st.session_state["derniere_usage"] = (usage.prompt_tokens, usage.completion_tokens)
        except Exception as e:
            diagnostiquer(e)
    question("3.3")


# ---- 3.2 ---------------------------------------------------------------------
def page_32():
    st.header("3.2 La température")
    st.markdown("Proche de 0, le modèle choisit presque toujours le token le plus probable ; plus elle augmente, "
                "plus les réponses varient. On pose **la même question plusieurs fois** et on compte les réponses distinctes.")
    prompt = champ_texte("Prompt", "w_p32", "Propose un nom original pour une startup tunisienne d'IA. Réponds par le nom seul.", hauteur=80)
    c1, c2 = st.columns(2)
    with c1:
        temps = multi("Températures à comparer", "w_T32", [0.0, 0.3, 0.7, 1.0, 1.5, 2.0], [0.0, 1.0])
    with c2:
        n = curseur("Nombre d'échantillons par température", "w_n32", 2, 10, 5)
    total = len(temps) * n
    st.caption(f"⚠️ Cela fera **{total} appels** à l'API (limite gratuite : ~30 requêtes/min).")
    if st.button("Lancer", type="primary", disabled=not temps):
        try:
            barre = st.progress(0.0)
            fait = 0
            for T in sorted(temps):
                reps = []
                for _ in range(n):
                    reps.append(demander(prompt, temperature=T, max_tokens=300)[0].strip())
                    fait += 1
                    barre.progress(fait / total)
                st.markdown(f"**T = {T}** → {len(set(reps))} réponse(s) distincte(s) sur {n}")
                st.write(reps)
            barre.empty()
        except Exception as e:
            diagnostiquer(e)
    question("3.2")


# ---- 3.4 ---------------------------------------------------------------------
def page_34():
    st.header("3.4 Qui compte les tokens ? API vs `tiktoken`")
    st.markdown("Chaque modèle a **son propre tokenizer**. Comparons le décompte de l'API avec celui de `tiktoken` "
                "(phrases de la page 1.2, modifiables ici aussi).")
    saisie = champ_texte("Phrases (`Langue | phrase`)", "w_phrases_langues", PHRASES_LANGUES, hauteur=140)
    paires = parse_libelle(saisie)
    st.caption(f"Cela fera {len(paires)} appel(s) à l'API.")
    if paires and st.button("Comparer", type="primary"):
        try:
            cl, o2 = get_enc("cl100k_base"), get_enc("o200k_base")
            rows = []
            for langue, phrase in paires:
                n_api = tokens_entree_api(phrase)
                n_cl, n_o2 = len(cl.encode(phrase)), len(o2.encode(phrase))
                rows.append({"Langue": langue, "API": n_api, "cl100k": n_cl, "o200k": n_o2, "API − o200k": n_api - n_o2})
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True)
            st.bar_chart(df.set_index("Langue")[["API", "cl100k", "o200k"]])
            ecarts = df["API − o200k"]
            st.info(f"Écart API − o200k : min {ecarts.min()}, max {ecarts.max()}. "
                    + ("Il est **constant**." if ecarts.nunique() == 1 else "Il **varie** selon les phrases."))
        except Exception as e:
            diagnostiquer(e)
    question("3.4")


# ---- 3.5 ---------------------------------------------------------------------
QUESTIONS_RAG = [
    "Quand ferme la bibliothèque le samedi ?",
    "Comment récupérer mon attestation de scolarité ?",
    "Quel est le prix du menu au restaurant universitaire ?",
]


def page_35():
    st.header("3.5 Mini-RAG : relier les trois briques")
    st.markdown("**RAG** : au lieu de compter sur la mémoire du modèle, on **retrouve** les passages pertinents "
                "(embeddings) puis on les **injecte dans le prompt** du LLM.")
    st.code("question → embedding → top-k passages → prompt = consignes + passages + question → LLM → réponse",
            language=None)
    docs = docs_courants() or lignes(DOCS_DEFAUT)
    st.caption(f"Base utilisée : {len(docs)} documents (modifiable page 2.3).")

    st.markdown("**Questions d'exemple :**")
    cols = st.columns(len(QUESTIONS_RAG))
    for col, q in zip(cols, QUESTIONS_RAG):
        col.button(q, key=f"exq35_{q}",
                   on_click=lambda q=q: st.session_state.update(w_q35=q))
    q = champ_ligne("Votre question", "w_q35", QUESTIONS_RAG[0])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k = curseur("k passages", "w_k35", 1, min(8, len(docs)), min(2, len(docs)))
    with c2:
        seuil = curseur("Score minimal", "w_seuil35", 0.0, 1.0, 0.0, step=0.05,
                        help="Les passages sous ce cosinus sont écartés.")
    with c3:
        T = curseur("Température", "w_T35", 0.0, 2.0, 0.0, step=0.1)
    with c4:
        mt = curseur("max_completion_tokens", "w_mt35", 100, 2000, 400)
    with st.expander("✏️ Consigne du prompt RAG (modifiable)"):
        consigne = champ_texte("Consigne", "w_consigne35", CONSIGNE_RAG, hauteur=100)
    comparer = case("Comparer avec la réponse **sans** contexte", "w_cmp35", True)

    if st.button("Poser la question", type="primary") and q.strip():
        X_docs = embeddings_phrases(tuple(docs))
        q_emb = embeddings_phrases((q,))[0]
        res = [(i, s) for i, s in top_k(q_emb, X_docs, k) if s >= seuil]
        passages = [docs[i] for i, _ in res]
        prompt = construire_prompt(q, passages, consigne)
        try:
            colA, colB = st.columns(2)
            if comparer:
                with colA:
                    st.markdown("### ❌ Sans contexte")
                    with st.spinner("…"):
                        texte0, u0, _ = demander(q, temperature=T, max_tokens=mt)
                    st.write(texte0 or "*(réponse vide)*")
                    st.caption(f"{u0.prompt_tokens} tokens en entrée · {u0.completion_tokens} en sortie")
            with (colB if comparer else st.container()):
                st.markdown("### ✅ Avec RAG")
                st.markdown("**Passages retrouvés :**" if res else "**Aucun passage au-dessus du seuil.**")
                for i, s in res:
                    st.markdown(f"- `{s:.2f}` — {docs[i]}")
                with st.spinner("…"):
                    texte, usage, fin = demander(prompt, temperature=T, max_tokens=mt)
                st.write(texte or "*(réponse vide)*")
                st.caption(f"{usage.prompt_tokens} tokens en entrée · {usage.completion_tokens} en sortie · "
                           f"finish_reason = {fin}")
                st.session_state["derniere_usage"] = (usage.prompt_tokens, usage.completion_tokens)
            with st.expander("👀 Prompt envoyé au LLM"):
                st.text(prompt)
        except Exception as e:
            diagnostiquer(e)
    question("3.5")


# ---- 3.6 ---------------------------------------------------------------------
def page_36():
    st.header("3.6 Coût et quotas")
    st.markdown(
        "Sur l'offre gratuite vous ne payez rien, mais vous êtes limité en **requêtes** et en **tokens** par minute "
        "et par jour. Le nombre de tokens (`usage`) est la « monnaie » commune : un **coût** en dollars sur une "
        "offre payante, un **quota** sur une offre gratuite.\n\n"
        "Cherchez les **tarifs actuels** du modèle (page des modèles / prix de Groq) et, dans la console, la page "
        "**Limits** de votre compte."
    )
    d_in, d_out = st.session_state.get("derniere_usage", (1500, 200))
    if "derniere_usage" in st.session_state:
        st.success(f"Valeurs préremplies avec votre dernier appel : {d_in} tokens en entrée, {d_out} en sortie.")
    c1, c2 = st.columns(2)
    with c1:
        n_in = nombre("Tokens en entrée (par question)", "w_nin36", int(d_in), min_value=0, step=50)
        p_in = nombre("Prix entrée ($ / million de tokens)", "w_pin36", 0.0, min_value=0.0, step=0.01, format="%.4f")
        n_q = nombre("Nombre de questions à estimer", "w_nq36", 10000, min_value=1, step=1000)
    with c2:
        n_out = nombre("Tokens en sortie (par question)", "w_nout36", int(d_out), min_value=0, step=50)
        p_out = nombre("Prix sortie ($ / million de tokens)", "w_pout36", 0.0, min_value=0.0, step=0.01, format="%.4f")
        quota = nombre("Quota de tokens par jour (console → Limits)", "w_quota36", 0, min_value=0, step=10000)
    c = cout_usd(n_in, n_out, p_in, p_out)
    a, b, d = st.columns(3)
    a.metric("Tokens par question", n_in + n_out)
    b.metric("Coût d'une question", f"{c:.6f} $")
    d.metric(f"Coût pour {n_q:,} questions".replace(",", " "), f"{c * n_q:.2f} $")
    if quota and (n_in + n_out) > 0:
        st.info(f"≈ {(n_in + n_out) / quota * 100:.2f} % du quota journalier pour UNE question "
                f"(environ {quota // (n_in + n_out)} questions par jour).")


# ---- Bilan / Rendu -----------------------------------------------------------
def page_bilan():
    st.header("🏁 Bilan")
    st.markdown("Dessinez (sur papier) le pipeline complet de votre mini-RAG, puis décrivez-le ci-dessous.")
    question("bilan")


def construire_rendu(nom):
    lignes_md = [f"# Lab 1 — Rendu de {nom or '(nom non renseigné)'}",
                 f"*Généré le {datetime.now():%d/%m/%Y à %H:%M}*", ""]
    for qid, (enonce, _) in QUESTIONS.items():
        rep = st.session_state.get(f"w_q_{qid}", "").strip()
        titre = "Question finale" if qid == "bilan" else f"Question {qid}"
        lignes_md += [f"## {titre}", f"> {enonce}", "", rep or "*(pas de réponse)*", ""]
    return "\n".join(lignes_md)


def page_rendu():
    st.header("📤 Rendu")
    st.markdown("Téléchargez vos réponses au format Markdown et déposez le fichier sur la plateforme de rendu. "
                "**Aucune clé API n'est incluse.**")
    nom = champ_ligne("Nom et prénom", "w_nom", "")
    repondues = sum(bool(st.session_state.get(f"w_q_{q}", "").strip()) for q in QUESTIONS)
    st.progress(repondues / len(QUESTIONS), text=f"{repondues} / {len(QUESTIONS)} questions traitées")
    contenu = construire_rendu(nom)
    fichier = "rendu_lab1" + (f"_{nom.strip().replace(' ', '_')}" if nom.strip() else "") + ".md"
    st.download_button("⬇️ Télécharger mes réponses", contenu, file_name=fichier, mime="text/markdown", type="primary")
    with st.expander("Aperçu"):
        st.markdown(contenu)


# ---- Espace enseignant -------------------------------------------------------
def page_enseignant():
    st.header("👩‍🏫 Espace enseignant")
    st.success("Mode enseignant actif : les réponses attendues apparaissent sous chaque question.")
    st.subheader("Barème (/20)")
    st.table(pd.DataFrame({
        "Partie": ["1", "2", "3", "Bonus"],
        "Détail": [
            "decouper (1) · fertilité + analyse (1,5) · pièges (1) · mini-BPE : paires/fusion (1), entraînement (1), encodage (0,5), analyse (1)",
            "cosine (0,5) · plus_proches (1) · matrice + voisin (1,5) · top_k (1) · PCA + limites (1) · questions (1)",
            "demander (1) · température (1) · tokens API vs tiktoken (1) · RAG (2,5) · coût (0,5) · bilan (1)",
            "courbe de compression, analogie, streaming",
        ],
        "Points": ["7", "6", "7", "+1 max"],
    }).set_index("Partie"))
    st.subheader("Logistique")
    st.markdown(
        "- Accès internet requis au premier usage (`tiktoken`, Hugging Face). En salle sans accès, pré-télécharger "
        "`gpt2` et `paraphrase-multilingual-MiniLM-L12-v2` et définir `TIKTOKEN_CACHE_DIR`.\n"
        "- **Une clé Groq par étudiant** (quotas par organisation et par modèle). Ne pas définir `GROQ_API_KEY` dans "
        "les secrets si vous voulez éviter une clé partagée saturée.\n"
        "- **Vérifier les modèles gratuits avant la séance** (page *Limits* de la console) ; sinon changer le modèle "
        "dans la page 3.0 (ex. `openai/gpt-oss-20b`).\n"
        "- Modèle de raisonnement : les tokens de réflexion comptent dans `max_completion_tokens`."
    )
    st.subheader("Points de discussion en plénière")
    st.markdown("Biais linguistiques des tokenizers (équité de coût entre langues) · hallucination et fiabilité · "
                "confidentialité des données envoyées à une API.")
    st.subheader("Toutes les réponses attendues")
    for qid, (enonce, rep) in QUESTIONS.items():
        with st.expander(("Question finale" if qid == "bilan" else f"Question {qid}") + f" — {enonce[:70]}…"):
            st.markdown(rep)


# ════════════════════════════════════════════════════════════════════════════
#  6. NAVIGATION
# ════════════════════════════════════════════════════════════════════════════
PAGES = {
    "🏠 Accueil": page_accueil,
    "1.1 Tokenizer réel": page_11,
    "1.2 Effet de la langue": page_12,
    "1.3 Pièges": page_13,
    "1.4 Mini-BPE": page_14,
    "2.1 Embeddings de tokens": page_21,
    "2.2 Embeddings de phrases": page_22,
    "2.3 Recherche sémantique": page_23,
    "2.5 Limites des embeddings": page_25,
    "3.0 Connexion API": page_30,
    "3.1 Bac à sable LLM": page_31,
    "3.2 Température": page_32,
    "3.4 Tokens API vs tiktoken": page_34,
    "3.5 Mini-RAG": page_35,
    "3.6 Coût et quotas": page_36,
    "🏁 Bilan": page_bilan,
    "📤 Rendu": page_rendu,
}


def mot_de_passe_enseignant():
    try:
        p = st.secrets.get("TEACHER_PASSWORD")
    except Exception:
        p = None
    return p or os.environ.get("TEACHER_PASSWORD")


def main():
    st.set_page_config(page_title="Lab 1 — Tokenisation, Embeddings, LLM", page_icon="🧠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # Conserve les widgets « w_* » quand on change de page
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("w_"):
            st.session_state[k] = st.session_state[k]

    # Mode enseignant (protégé par mot de passe)
    attendu = mot_de_passe_enseignant()
    with st.sidebar.expander("🔒 Enseignant"):
        if attendu:
            st.session_state.setdefault("w_pwd", "")
            saisi = st.text_input("Mot de passe", type="password", key="w_pwd")
            st.session_state["_enseignant"] = bool(saisi) and hmac.compare_digest(saisi, attendu)
        else:
            st.session_state["_enseignant"] = False
            st.caption("Définissez `TEACHER_PASSWORD` (secrets Streamlit ou variable d'environnement) "
                       "pour activer ce mode.")

    pages = dict(PAGES)
    if est_enseignant():
        pages["👩‍🏫 Espace enseignant"] = page_enseignant

    st.sidebar.title("🧠 Lab 1")
    st.session_state.setdefault("w_page", "🏠 Accueil")
    if st.session_state["w_page"] not in pages:
        st.session_state["w_page"] = "🏠 Accueil"
    choix = st.sidebar.radio("Navigation", list(pages), key="w_page", label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("🔑 Clé API : " + ("✅ détectée" if lire_cle() else "❌ non renseignée (page 3.0)"))

    pages[choix]()


if __name__ == "__main__":
    main()
