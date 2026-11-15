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


def _passe1(transcription: str, incertains: list | None = None) -> str:
    return json.dumps({"incertains": incertains or [], "transcription": transcription})


def test_deux_passes_nominal(monkeypatch) -> None:
    incertains = [{"mot": "soire", "lettres_vues": "s-o-i-r-e", "variantes": ["soir"]}]
    scorer = _make_scorer_deux_passes(
        [_passe1("Le soire[?] tombait", incertains), _CODAGE_OK], monkeypatch
    )
    pred = scorer.score_copy(_copy(), "Le soir tombait")

    appels = scorer.client.appels
    # Passe 1 (image) puis codage : l'alignement est local, sans appel LLM.
    assert len(appels) == 2
    assert "Le soir tombait" not in json.dumps(appels[0]["messages"], ensure_ascii=False)
    assert appels[0]["messages"][-1]["content"][1]["type"] == "image_url"
    texte_codage = json.dumps(appels[1]["messages"], ensure_ascii=False)
    assert "« soir » (mot) | écrit : « soire »" in texte_codage

    assert pred.raw_transcription == "Le soire[?] tombait"
    assert pred.transcription_passe1 == "Le soire[?] tombait"
    assert [it.transcription for it in pred.items] == ["Le", "soire", "tombait"]
    assert pred.n_incertains_passe1 == 1
    assert pred.n_absents_alignement == 0
    assert pred.n_mots_en_trop_alignement == 0
    assert pred.n_attempts == 1


def test_deux_passes_mot_oublie_ne_decale_pas(monkeypatch) -> None:
    scorer = _make_scorer_deux_passes([_passe1("Le tombait"), _CODAGE_OK], monkeypatch)
    pred = scorer.score_copy(_copy(), "Le soir tombait")
    assert "« soir » (mot) | écrit : (rien)" in json.dumps(
        scorer.client.appels[1]["messages"], ensure_ascii=False
    )
    assert [it.transcription for it in pred.items] == ["Le", "", "tombait"]
    assert pred.n_absents_alignement == 1


def test_deux_passes_echec_passe1(monkeypatch) -> None:
    # max_retries=1 → deux essais de passe 1, réponses vides.
    scorer = _make_scorer_deux_passes(["", ""], monkeypatch)
    pred = scorer.score_copy(_copy(), "Le soir tombait")
    assert pred.transcribed is False
    assert pred.n_attempts == 2


def test_prompt_passe1_consigne_ratures() -> None:
    from evaluation_dictee.pipeline.prompts import (
        _CONSIGNE_RATURES_TRANSCRIPTION,
        build_transcription_passe1_prompt,
    )

    avec = build_transcription_passe1_prompt(read_final_state=True)[0]["content"]
    sans = build_transcription_passe1_prompt(read_final_state=False)[0]["content"]
    assert "Ratures : " + _CONSIGNE_RATURES_TRANSCRIPTION in avec
    assert "Ratures" not in sans
    assert "{{" not in avec + sans


def test_prompt_passe1_ronds_sur_i_et_majuscule_apres_point() -> None:
    from evaluation_dictee.pipeline.prompts import build_transcription_passe1_prompt

    systeme = build_transcription_passe1_prompt()[0]["content"]
    assert "rond au lieu du point sur le i" in systeme
    assert "Après chaque point" in systeme


def test_prompt_passe1_regles_elisions() -> None:
    from evaluation_dictee.pipeline.prompts import build_transcription_passe1_prompt

    sans = build_transcription_passe1_prompt()[0]["content"]
    avec = build_transcription_passe1_prompt(regles_elisions=True)[0]["content"]
    assert "jambages" not in sans and "{{" not in sans
    assert "jambages" in avec and "lettre suivie d'une apostrophe" in avec
    # Règles ajoutées après la règle 9, avant la consigne des ratures.
    assert avec.index("9 - Après chaque point") < avec.index("10 - ") < avec.index("Ratures")


def test_passe1_tronquee_relancee(monkeypatch) -> None:
    """Une passe 1 coupée par max_tokens est relancée, jamais prise comme texte brut."""
    scorer = _make_scorer_deux_passes([], monkeypatch)
    reponses = ['{"incertains": [], "transcription": "Le Le Le Le', _passe1("Le soir tombait")]
    fins = ["length", "stop"]
    appels: list[dict] = []

    def _create(**kwargs):  # noqa: ANN003
        appels.append(kwargs)
        msg = SimpleNamespace(content=reponses.pop(0), model_extra={})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason=fins.pop(0))],
            usage=SimpleNamespace(completion_tokens=100),
        )

    scorer.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    transcription, n, passe1, _ = scorer._transcribe_et_aligner(_copy())
    assert transcription == "Le soir tombait"
    assert n == 2


def test_vote_entre_trois_lecteurs(monkeypatch) -> None:
    """Chaque lecteur lit puis fait coder ; le code retenu est majoritaire par item."""
    scorer = _make_scorer_deux_passes([], monkeypatch)
    scorer.lecteurs_supplementaires = [ModelConfig(name="m2"), ModelConfig(name="m3")]

    def codage(codes: list[str], lus: list[str]) -> str:
        return json.dumps(
            {
                "items": [
                    {"item_id": f"i{k + 1}", "transcription": t, "code": c}
                    for k, (c, t) in enumerate(zip(codes, lus, strict=True))
                ]
            }
        )

    scorer.client = _FakeClient(
        [
            _passe1("Le soir tombait"),
            codage(["1", "1", "1"], ["Le", "soir", "tombait"]),
            _passe1("Le soire tombait"),
            codage(["1", "9", "1"], ["Le", "soire", "tombait"]),
            _passe1("Le soire tombé"),
            codage(["1", "9", "9"], ["Le", "soire", "tombé"]),
        ]
    )
    pred = scorer.score_copy(_copy(), "Le soir tombait")

    appels = scorer.client.appels
    assert [a["model"] for a in appels[::2]] == ["m1", "m2", "m3"]  # passe 1 de chaque lecteur
    assert [it.code for it in pred.items] == ["1", "9", "1"]
    assert [round(it.confidence, 2) for it in pred.items] == [1.0, 0.67, 0.67]
    assert pred.items[1].codes_lecteurs == ["1", "9", "9"]
    # Transcription de l'item : celle d'un lecteur qui a voté le code retenu.
    assert pred.items[1].transcription == "soire"
    assert pred.raw_transcription == "Le soir tombait"  # lecteur principal
    assert pred.transcriptions_lecteurs == [
        "Le soir tombait",
        "Le soire tombait",
        "Le soire tombé",
    ]
    assert pred.n_attempts == 3


def test_extra_readers_exige_alignement() -> None:
    import pytest

    from evaluation_dictee.config import ExperimentConfig, run_output_name

    base = {
        "name": "r",
        "approach": "two_stage",
        "model": {"name": "a"},
        "model_stage2": {"name": "c"},
        "extra_readers": [{"name": "b"}],
        "data": {"corpus": "dictee", "images_path": "x", "labels_path": "y", "grid_path": "z"},
    }
    with pytest.raises(ValueError, match="extra_readers"):
        ExperimentConfig.model_validate(base)
    cfg = ExperimentConfig.model_validate({**base, "prompt": {"transcription_verification": True}})
    assert run_output_name(cfg) == "r_a_c_b"
