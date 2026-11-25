"""Tests unitaires de la logique pure d'alignement (sans SDK, exécutables partout)."""

from __future__ import annotations

import sys
import types
import pathlib


def _stub_pydantic():
    """Pydantic minimal : champ -> attribut, défauts, et model_validator(mode="after")."""
    if "pydantic" in sys.modules:
        return  # déjà un vrai pydantic sur cette machine

    pyd = types.ModuleType("pydantic")

    def Field(default=None, **kwargs):
        return default

    def model_validator(*args, mode="after", **kwargs):
        def deco(fn):
            fn.__is_model_validator__ = True
            return fn

        return deco

    class BaseModel:
        def __init__(self, **data):
            for name in getattr(type(self), "__annotations__", {}):
                if name in data:
                    setattr(self, name, data.pop(name))
                else:
                    setattr(self, name, getattr(type(self), name, None))
            if data:
                raise TypeError(f"Champs inconnus: {sorted(data)}")
            for attr_name in dir(type(self)):
                attr = getattr(type(self), attr_name)
                if getattr(attr, "__is_model_validator__", False):
                    attr(self)

    pyd.BaseModel = BaseModel
    pyd.Field = Field
    pyd.model_validator = model_validator
    sys.modules["pydantic"] = pyd


def _load_module():
    """Charge pronunciation.py en stubbant le SDK mistralai (non installé en local)."""
    _stub_pydantic()
    for name in (
        "mistralai",
        "mistralai.workflows",
        "mistralai.workflows.plugins",
        "mistralai.workflows.plugins.mistralai",
    ):
        sys.modules.pop(name, None)

    workflows_mod = types.ModuleType("mistralai.workflows")

    class _Activity:
        def __call__(self, fn=None):
            if fn is None:
                return lambda f: f
            return fn

    class _Define:
        def __call__(self, **kwargs):
            def deco(cls):
                return cls

            return deco

    ns = types.SimpleNamespace()
    workflows_mod.activity = _Activity()
    workflows_mod.workflow = ns
    ns.define = _Define()
    ns.entrypoint = lambda fn: fn

    mistralai_mod = types.ModuleType("mistralai")
    mistralai_mod.workflows = workflows_mod
    plugins_mod = types.ModuleType("mistralai.workflows.plugins")
    mistralai_plugins = types.ModuleType("mistralai.workflows.plugins.mistralai")
    mistralai_plugins.get_mistral_client = lambda: None
    plugins_mod.mistralai = mistralai_plugins
    workflows_mod.plugins = plugins_mod

    sys.modules["mistralai"] = mistralai_mod
    sys.modules["mistralai.workflows"] = workflows_mod
    sys.modules["mistralai.workflows.plugins"] = plugins_mod
    sys.modules["mistralai.workflows.plugins.mistralai"] = mistralai_plugins

    import importlib.util

    path = (
        pathlib.Path(__file__).parent.parent
        / "src"
        / "worker"
        / "workflows"
        / "pronunciation.py"
    )
    spec = importlib.util.spec_from_file_location("pronunciation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perfect_match():
    mod = _load_module()
    ops = mod.align_words(
        "I think this is the third month.", "I think this is the third month."
    )
    assert all(status == "ok" for status, _, _ in ops)


def test_th_error_think_to_sink():
    mod = _load_module()
    ops = mod.align_words(
        "I think this is the third month.", "I sink this is the third month"
    )
    subs = [op for op in ops if op[0] == "sub"]
    assert len(subs) == 1
    assert subs[0][1] == "think" and subs[0][2] == "sink"


def test_missing_word():
    mod = _load_module()
    ops = mod.align_words("The sheep is sleeping on the ship.", "The sheep is on the ship")
    missing = [op for op in ops if op[0] == "missing"]
    assert missing and missing[0][1] == "sleeping"


def test_extra_word():
    mod = _load_module()
    ops = mod.align_words("She finished her work.", "She finished her work and smiled")
    extras = [op for op in ops if op[0] == "extra"]
    assert extras and extras[0][1] == "and"


def test_dropped_h():
    mod = _load_module()
    ops = mod.align_words("He has a hard hat.", "He as a hard hat.")
    subs = [op for op in ops if op[0] == "sub"]
    assert any(op[1] == "has" and op[2] == "as" for op in subs)


def test_close_word_counts_three_quarters():
    mod = _load_module()
    ops = mod.align_words("The weather is better than ever.", "The weather is better then ever")
    closes = [op for op in ops if op[0] == "close"]
    assert any(op[1] == "than" and op[2] == "then" for op in closes)


def test_input_requires_audio():
    mod = _load_module()
    try:
        mod.PronunciationInput(target_phrase="hello")
    except Exception:
        return  # attendu : audio_base64 ou audio_url obligatoire
    raise AssertionError("PronunciationInput doit exiger audio_base64 ou audio_url")


def test_tip_for_sub_leverages_word_tips():
    mod = _load_module()
    tip = mod._tip_for("think", "sub", "sink")
    assert "sink" in tip and "langue" in tip


if __name__ == "__main__":
    for test_name, test_fn in sorted(
        (k, v) for k, v in globals().items() if k.startswith("test_")
    ):
        test_fn()
        print(f"OK  {test_name}")
    print("ALL PURE TESTS PASSED")
