"""Tests du TwoStageScorer (approche 1 : transcription puis codage)."""

import json
from types import SimpleNamespace

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.data.loaders import Copy
from evaluation_dictee.models.two_stage import TwoStageScorer

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", [], [], []),
    GridItem("i3", "tombait", "mot", [], [], []),
]


class _FakeClient:
    """Client OpenAI factice : renvoie des réponses scriptées dans l'ordre."""

    def __init__(self, reponses: list[str]) -> None:
        self._reponses = list(reponses)
        self.appels: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):  # noqa: ANN003
        self.appels.append(kwargs)
        contenu = self._reponses.pop(0)
        msg = SimpleNamespace(content=contenu, model_extra={})
        # `finish_reason` fait partie de toute réponse réelle : le pipeline s'en sert
        # pour détecter une génération tronquée par `max_tokens`.
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(completion_tokens=100),
        )


def _make_scorer(reponses: list[str], monkeypatch) -> TwoStageScorer:
    # Évite le vrai chargement d'image
    monkeypatch.setattr("evaluation_dictee.models.two_stage.load_image", lambda path: "FAKE_IMAGE")
    monkeypatch.setattr(
        "evaluation_dictee.models.two_stage._image_to_data_url", lambda img: "data:fake"
    )
    scorer = TwoStageScorer(
        model_config=ModelConfig(name="m1", max_retries=1),
        prompt_config=PromptConfig(),
        base_url="http://x",
        api_key="k",
        grid_items=ITEMS,
    )
    scorer.client = _FakeClient(reponses)
    return scorer


def _copy() -> Copy:
    return Copy(
        copy_id="c.png",
        image_path="/tmp/c.png",
        item_ids=["i1", "i2", "i3"],
        expert_codes=["1", "1", "1"],
    )


def test_deux_etapes_nominal(monkeypatch) -> None:
    transcription = json.dumps({"transcription": "Le soir tombait"})
    codage = json.dumps(
        {
            "items": [
                {"item_id": "i1", "transcription": "Le", "code": "1", "confidence": 0.9},
                {"item_id": "i2", "transcription": "soir", "code": "1", "confidence": 0.9},
                {"item_id": "i3", "transcription": "tombait", "code": "1", "confidence": 0.9},
            ]
        }
    )
    scorer = _make_scorer([transcription, codage], monkeypatch)
    pred = scorer.score_copy(_copy(), "Le soir tombait")
    assert pred.transcribed is True
    assert len(pred.items) == 3
    assert [it.code for it in pred.items] == ["1", "1", "1"]


def test_deux_appels_distincts(monkeypatch) -> None:
    """Vérifie qu'il y a bien DEUX appels : étape 1 (image) puis étape 2 (texte)."""
    transcription = json.dumps({"transcription": "Le soir tombait"})
    codage = json.dumps(
        {
            "items": [
                {"item_id": "i1", "transcription": "Le", "code": "1", "confidence": 0.9},
                {"item_id": "i2", "transcription": "soir", "code": "1", "confidence": 0.9},
                {"item_id": "i3", "transcription": "tombait", "code": "1", "confidence": 0.9},
            ]
        }
    )
    scorer = _make_scorer([transcription, codage], monkeypatch)
    scorer.score_copy(_copy(), "Le soir tombait")
    assert len(scorer.client.appels) == 2
    # Chaque prompt est désormais structuré system + user ; l'image (ou le texte)
    # est portée par le message user, c.-à-d. le dernier message.
    # 1er appel = multimodal : le contenu du message user est une liste (avec image).
    contenu1 = scorer.client.appels[0]["messages"][-1]["content"]
    assert isinstance(contenu1, list)
    # 2e appel = texte seul : le contenu du message user est une chaîne.
    contenu2 = scorer.client.appels[1]["messages"][-1]["content"]
    assert isinstance(contenu2, str)


def test_transcription_vide_marque_non_transcrite(monkeypatch) -> None:
    # L'étape 1 échoue à toutes les tentatives → copie non transcrite
    vide = json.dumps({"transcription": ""})
    scorer = _make_scorer([vide, vide], monkeypatch)  # max_retries=1 → 2 essais
    pred = scorer.score_copy(_copy(), "Le soir tombait")
    assert pred.transcribed is False
    # Pas d'appel à l'étape 2 si pas de transcription
    assert len(scorer.client.appels) == 2  # 2 tentatives étape 1, 0 étape 2


def test_modele_stage2_distinct(monkeypatch) -> None:
    """Le modèle de l'étape 2 doit être utilisé pour le second appel."""
    monkeypatch.setattr("evaluation_dictee.models.two_stage.load_image", lambda path: "IMG")
    monkeypatch.setattr(
        "evaluation_dictee.models.two_stage._image_to_data_url", lambda img: "data:x"
    )
    scorer = TwoStageScorer(
        model_config=ModelConfig(name="htr-model"),
        prompt_config=PromptConfig(),
        base_url="http://x",
        api_key="k",
        grid_items=ITEMS,
        model_config_stage2=ModelConfig(name="text-model", kind="llm"),
    )
    transcription = json.dumps({"transcription": "Le soir tombait"})
    codage = json.dumps(
        {
            "items": [
                {"item_id": "i1", "transcription": "Le", "code": "1", "confidence": 0.9},
                {"item_id": "i2", "transcription": "soir", "code": "1", "confidence": 0.9},
                {"item_id": "i3", "transcription": "tombait", "code": "1", "confidence": 0.9},
            ]
        }
    )
    scorer.client = _FakeClient([transcription, codage])
    scorer.score_copy(_copy(), "Le soir tombait")
    assert scorer.client.appels[0]["model"] == "htr-model"
    assert scorer.client.appels[1]["model"] == "text-model"


def test_parse_transcription_repli_texte_brut() -> None:
    # Si le modèle répond en texte brut (pas de JSON), on récupère quand même
    assert TwoStageScorer._parse_transcription("Le soir tombait") == "Le soir tombait"


# ── Transcription en deux passes (option transcription_verification) ─────────


def _make_scorer_deux_passes(reponses: list[str], monkeypatch) -> TwoStageScorer:
    scorer = _make_scorer(reponses, monkeypatch)
    scorer.prompt_config = PromptConfig(transcription_verification=True)
    return scorer


_CODAGE_OK = json.dumps(
    {
        "items": [
            {"item_id": "i1", "transcription": "Le", "code": "1"},
            {"item_id": "i2", "transcription": "soire", "code": "9"},
            {"item_id": "i3", "transcription": "tombait", "code": "1"},
        ]
    }
)


_PASSE1_EPELEE = json.dumps(
    {
        "lignes": [["L e", "s o i r [?]", "t o m b a i t"]],
        "incertains": [
            {"ligne": 1, "position": 2, "lettres_vues": "s-o-i-r-?", "variantes": ["s o i r e"]}
        ],
    }
)


def _passe2_choisit(scorer: TwoStageScorer, cible: str) -> str:
    """Réponse de passe 2 qui choisit `cible` pour le mot « soir », la passe 1 ailleurs."""
    # Reconstruit les questions avec la même graine que le scorer (id de la copie).
    from evaluation_dictee.pipeline.transcription_epelee import (
        construire_questions,
        lignes_depuis_reponse,
    )

    lignes = lignes_depuis_reponse(json.loads(_PASSE1_EPELEE))
    assert lignes is not None
    questions = construire_questions(
        lignes, ITEMS, json.loads(_PASSE1_EPELEE)["incertains"], graine="c.png"
    )
    reponses = []
    for q in questions:
        mot = cible if q.lu == "soir" else q.lu
        lettre = "ABCDEFGH"[q.options.index(mot)]
        reponses.append(
            {
                "id": q.id,
                "lettres_vues": "-",
                "choix": lettre,
                "epele": " ".join(mot),
                "certain": True,
            }
        )
    return json.dumps({"reponses": reponses})


def test_deux_passes_nominal(monkeypatch) -> None:
    scorer = _make_scorer_deux_passes([], monkeypatch)
    scorer.client = _FakeClient([_PASSE1_EPELEE, _passe2_choisit(scorer, "soire"), _CODAGE_OK])
    pred = scorer.score_copy(_copy(), "Le soir tombait")

    appels = scorer.client.appels
    assert len(appels) == 3
    # Aucune des deux passes ne voit le texte de référence.
    for appel in appels[:2]:
        assert "Le soir tombait" not in json.dumps(appel["messages"], ensure_ascii=False)
    # Passe 2 : image + questions fermées construites depuis la passe 1.
    user_p2 = appels[1]["messages"][-1]["content"]
    assert user_p2[1]["type"] == "image_url"
    assert "Question 1" in user_p2[0]["text"]
    assert "soire" in user_p2[0]["text"]
    # Le codage part de la transcription issue des choix de la passe 2.
    assert "Le soire tombait" in json.dumps(appels[2]["messages"], ensure_ascii=False)

    assert pred.raw_transcription == "Le soire tombait"
    assert pred.transcription_passe1 == "Le soir[?] tombait"
    assert pred.n_corrections_passe2 == 1
    assert pred.n_incertains_passe1 == 1
    assert pred.n_incertains_passe2 == 0
    assert pred.n_attempts == 2


def test_deux_passes_echec_passe2_garde_passe1(monkeypatch) -> None:
    passe1 = json.dumps({"lignes": [["L e", "s o i r e", "t o m b a i t"]], "incertains": []})
    # max_retries=1 → deux essais de passe 2, tous deux illisibles.
    scorer = _make_scorer_deux_passes(
        [passe1, "pas du json", "toujours pas", _CODAGE_OK], monkeypatch
    )
    pred = scorer.score_copy(_copy(), "Le soir tombait")
    assert pred.raw_transcription == "Le soire tombait"
    assert pred.transcription_passe1 == "Le soire tombait"
    assert pred.n_corrections_passe2 is None
    assert pred.n_attempts == 3


def test_consigne_ratures_identique_dans_les_deux_passes() -> None:
    from evaluation_dictee.pipeline.prompts import (
        _CORPS_RATURES_EPELE,
        build_transcription_passe1_prompt,
        build_transcription_passe2_prompt,
    )

    p1 = build_transcription_passe1_prompt(read_final_state=True)[0]["content"]
    p2 = build_transcription_passe2_prompt("Q", read_final_state=True)[0]["content"]
    assert "7 - " + _CORPS_RATURES_EPELE in p1
    assert "6 - " + _CORPS_RATURES_EPELE in p2
    assert "{{" not in p1 + p2
