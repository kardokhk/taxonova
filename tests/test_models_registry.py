"""Model registry: id and alias resolution, the models directory, downloads
with URL fallback and checksum verification, and the compatibility layer in
taxonova.utils.model_downloader. No network access: downloads use file://
URLs or a local HTTP server.
"""
import functools
import hashlib
import http.server
import threading
from pathlib import Path

import pytest

from taxonova import models
from taxonova.models import ModelSpec, UnknownModelError, ensure_model, get_model, list_models

# --- resolution ------------------------------------------------------------

EXPECTED = {
    "v4": ["v4", "V4", "gg2_v4", "GG2_V4", "gg2_v4_mrgroups_model_26.10.pt",
           "taxonova-V4-greengenes2-26.10.pt"],
    "v3v4": ["v3v4", "V3V4", "V3-V4", "v3-v4", "gg2_v3v4", "gg2_v3v4_mrgroups_model_26.10.pt",
             "taxonova-V3V4-greengenes2-26.10.pt"],
    "v5v7": ["v5v7", "V5-V7", "gg2_v5v7", "gg2_v5v7_mrgroups_model_26.10.pt",
             "taxonova-V5V7-greengenes2-26.10.pt"],
    "16s": ["16s", "16S", "v1v9", "V1-V9", "full", "gg2_v1v9", "gg2_v1v9_mrgroups_model_26.10.pt",
            "taxonova-16S-greengenes2-26.10.pt"],
    "operon": ["operon", "rrn", "RRN", "gtdb_rrn", "gtdb_rrn_mrgroups_model.pt",
               "taxonova-operon-grond-1.0.pt"],
}


@pytest.mark.parametrize("model_id,name", [(m, n) for m, names in EXPECTED.items() for n in names])
def test_every_alias_resolves(model_id, name):
    assert get_model(name).id == model_id
    assert get_model(f"  {name} ").id == model_id


def test_registry_contents():
    specs = list_models()
    assert [s.id for s in specs] == ["v4", "v3v4", "v5v7", "16s", "operon"]
    names = set()
    for s in specs:
        assert s.status == "stable"
        assert len(s.sha256) == 64 and len(s.md5) == 32
        assert s.filename.startswith("taxonova-") and s.filename.endswith(".pt")
        assert "mrgroups" not in s.filename
        assert s.urls[0] == f"https://github.com/kardokhk/taxonova/releases/download/v26.10.1/{s.filename}"
        assert s.urls[1] == f"https://huggingface.co/claessonlab/taxonova-models/resolve/main/{s.filename}"
        for n in (s.id, s.filename) + s.aliases:
            assert n.lower() not in names, f"name {n} used twice"
            names.add(n.lower())
    assert get_model("operon").sha256 == "9ac36fea212edd569198027c9e6fb182199bb316acacad188d758929c9408289"


def test_checksums_unchanged_from_26_10_0():
    # Same bytes as the 26.10.0 / v1.0.0 assets under their earlier names.
    md5 = {"v4": "bccf86c9086b68defe8bc8a2a3dff2b6", "v3v4": "a5e584504627a8055afe25fd56893e37",
           "v5v7": "6de1add678c05fd0c5c3679ea4094d1d", "16s": "103e508682ad0b948e9f8a993bb7be0a",
           "operon": "6f5d4ef039888a33ad765ef41725b624"}
    for model_id, value in md5.items():
        assert get_model(model_id).md5 == value


def test_unknown_model_lists_valid_ids():
    with pytest.raises(KeyError) as err:
        get_model("silva")
    text = str(err.value)
    assert isinstance(err.value, UnknownModelError)
    assert "v4, v3v4, v5v7, 16s, operon" in text and not text.startswith("'")


def test_superseded_v1_names_are_not_aliases():
    for name in models.SUPERSEDED_MODEL_FILES:
        with pytest.raises(KeyError, match="replaced in taxonova 26.10.0"):
            get_model(name)


def test_preview_models_hidden_by_default(monkeypatch):
    preview = ModelSpec(id="v4-silva", name="16S V4, SILVA", region="16S V4", reference="SILVA",
                        reference_version="138.2", taxonomy="SILVA", model_version="26.11",
                        filename="taxonova-V4-silva-26.11.pt", urls=("file:///nonexistent",),
                        sha256="0" * 64, status="preview")
    monkeypatch.setattr(models, "_REGISTRY", models._REGISTRY + [preview])
    assert "v4-silva" not in [s.id for s in list_models()]
    assert "v4-silva" in [s.id for s in list_models(include_hidden=True)]
    assert get_model("V4-SILVA") is preview


def test_models_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TAXONOVA_MODELS_DIR", str(tmp_path / "m"))
    assert models.models_dir() == tmp_path / "m"
    monkeypatch.delenv("TAXONOVA_MODELS_DIR")
    assert models.models_dir() == Path.home() / ".taxonova" / "models"


# --- downloads -------------------------------------------------------------

PAYLOAD = b"taxonova test checkpoint " * 1000


def _fake(urls, payload=PAYLOAD, **kw):
    return ModelSpec(id="fake", name="fake", region="16S V4", reference="test", reference_version="",
                     taxonomy="", model_version="0", filename="taxonova-fake-0.pt", urls=tuple(urls),
                     sha256=hashlib.sha256(payload).hexdigest(), md5=hashlib.md5(payload).hexdigest(),
                     size_mb=0, aliases=("old_fake_name.pt",), **kw)


@pytest.fixture
def source(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "taxonova-fake-0.pt").write_bytes(PAYLOAD)
    (d / "wrong.pt").write_bytes(b"not the published bytes")
    return d


@pytest.fixture
def http_source(source):
    handler = functools.partial(_QuietHandler, directory=str(source))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _install(monkeypatch, spec):
    monkeypatch.setattr(models, "_REGISTRY", models._REGISTRY + [spec])
    return spec


def test_download_file_url_then_reuse(monkeypatch, tmp_path, source, capsys):
    spec = _install(monkeypatch, _fake([(source / "taxonova-fake-0.pt").as_uri()]))
    dest = tmp_path / "models"
    path = ensure_model("fake", models_dir=dest)
    assert path == dest / spec.filename and path.read_bytes() == PAYLOAD
    assert not list(dest.glob("*.part"))
    err = capsys.readouterr().err
    assert "not downloaded yet" in err and "Downloading model fake" in err
    mtime = path.stat().st_mtime_ns
    assert ensure_model("FAKE", models_dir=dest) == path
    assert path.stat().st_mtime_ns == mtime
    assert capsys.readouterr().err == ""


def test_http_fallback_to_second_url(monkeypatch, tmp_path, http_source, capsys):
    spec = _install(monkeypatch, _fake([f"{http_source}/missing.pt", f"{http_source}/taxonova-fake-0.pt"]))
    path = ensure_model("fake", models_dir=tmp_path / "m")
    assert path.read_bytes() == PAYLOAD
    err = capsys.readouterr().err
    assert "could not use 127.0.0.1" in err and "404" in err


def test_wrong_bytes_at_first_url_falls_back(monkeypatch, tmp_path, http_source):
    _install(monkeypatch, _fake([f"{http_source}/wrong.pt", f"{http_source}/taxonova-fake-0.pt"]))
    assert ensure_model("fake", models_dir=tmp_path / "m").read_bytes() == PAYLOAD


def test_all_urls_fail(monkeypatch, tmp_path, http_source):
    _install(monkeypatch, _fake([f"{http_source}/wrong.pt", f"{http_source}/missing.pt"]))
    dest = tmp_path / "m"
    with pytest.raises(RuntimeError, match="Could not download model fake") as err:
        ensure_model("fake", models_dir=dest)
    assert "wrong.pt" in str(err.value) and "missing.pt" in str(err.value)
    assert not (dest / "taxonova-fake-0.pt").exists()
    assert not list(dest.glob("*.part"))


def test_sha256_alone_rejects(monkeypatch, tmp_path, source):
    spec = _fake([(source / "taxonova-fake-0.pt").as_uri()])
    bad = ModelSpec(**{**spec.__dict__, "sha256": "0" * 64})
    _install(monkeypatch, bad)
    with pytest.raises(RuntimeError, match="checksum"):
        ensure_model("fake", models_dir=tmp_path / "m")


def test_mismatched_cache_is_reported_and_redownloaded(monkeypatch, tmp_path, source, capsys):
    spec = _install(monkeypatch, _fake([(source / "taxonova-fake-0.pt").as_uri()]))
    dest = tmp_path / "m"
    dest.mkdir()
    (dest / spec.filename).write_bytes(b"an older checkpoint under the same name")
    path = ensure_model("fake", models_dir=dest)
    assert path.read_bytes() == PAYLOAD
    assert "does not match the published checksum" in capsys.readouterr().err


def test_force_downloads_again(monkeypatch, tmp_path, source, capsys):
    _install(monkeypatch, _fake([(source / "taxonova-fake-0.pt").as_uri()]))
    dest = tmp_path / "m"
    ensure_model("fake", models_dir=dest)
    capsys.readouterr()
    ensure_model("fake", models_dir=dest, force=True)
    assert "Downloading model fake" in capsys.readouterr().err


def test_copy_under_earlier_file_name_is_used(monkeypatch, tmp_path):
    spec = _install(monkeypatch, _fake(["file:///nonexistent/never-fetched.pt"]))
    dest = tmp_path / "m"
    dest.mkdir()
    (dest / "old_fake_name.pt").write_bytes(PAYLOAD)
    assert ensure_model("fake", models_dir=dest) == dest / "old_fake_name.pt"
    assert models.find_cached(spec, dest) == dest / "old_fake_name.pt"


def test_quiet_prints_nothing(monkeypatch, tmp_path, source, capsys):
    _install(monkeypatch, _fake([(source / "taxonova-fake-0.pt").as_uri()]))
    ensure_model("fake", models_dir=tmp_path / "m", quiet=True)
    assert capsys.readouterr().err == ""


def test_progress_is_silent_off_terminal(monkeypatch, tmp_path, http_source, capsys):
    # Regression for the 26.10.0 downloader, which printed a line per 8 KB block.
    _install(monkeypatch, _fake([f"{http_source}/taxonova-fake-0.pt"]))
    ensure_model("fake", models_dir=tmp_path / "m")
    err = capsys.readouterr().err
    assert "\r" not in err and err.count("\n") == 3


def test_existing_path_is_returned_unchanged(tmp_path):
    own = tmp_path / "my_model.pt"
    own.write_bytes(b"x")
    assert ensure_model(str(own)) == own


def test_bare_file_name_in_models_dir(tmp_path):
    (tmp_path / "my_model.pt").write_bytes(b"x")
    assert ensure_model("my_model.pt", models_dir=tmp_path) == tmp_path / "my_model.pt"


def test_missing_file_and_unknown_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="replaced in taxonova 26.10.0"):
        ensure_model("gg2_v4_mrgroups_model.pt", models_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        ensure_model(str(tmp_path / "nope.pt"), models_dir=tmp_path)
    with pytest.raises(KeyError):
        ensure_model("silva", models_dir=tmp_path)


def test_superseded_warning_needs_name_and_md5(tmp_path, monkeypatch):
    payload = b"pretend v1.0.0 checkpoint"
    name = "gg2_v4_mrgroups_model.pt"
    monkeypatch.setitem(models.SUPERSEDED_MODEL_FILES, name, ("v4", hashlib.md5(payload).hexdigest()))
    old = tmp_path / name
    old.write_bytes(payload)
    msg = models.superseded_model_warning(old)
    assert msg and "model v4" in msg and "gg2_v4_mrgroups_model_26.10.pt" in msg
    other = tmp_path / "x" / name
    other.parent.mkdir()
    other.write_bytes(b"a different file under the old name")
    assert models.superseded_model_warning(other) is None


# --- compatibility layer -----------------------------------------------------

def test_model_downloader_shim(tmp_path, monkeypatch, source):
    from taxonova.utils import model_downloader as md
    assert list(md.AVAILABLE_MODELS) == ["gg2_v4", "gg2_v3v4", "gg2_v5v7", "gg2_v1v9", "gtdb_rrn"]
    for key, info in md.AVAILABLE_MODELS.items():
        spec = get_model(key)
        assert info["filename"] == spec.filename and info["url"] == spec.urls[0]
        assert info["sha256"] == spec.sha256 and info["md5"] == spec.md5
        for field in ("name", "description", "size_mb"):
            assert info[field]
    monkeypatch.setenv("TAXONOVA_MODELS_DIR", str(tmp_path / "env"))
    assert md.ModelDownloader().models_dir == tmp_path / "env"

    _install(monkeypatch, _fake([(source / "taxonova-fake-0.pt").as_uri()]))
    dl = md.ModelDownloader(models_dir=tmp_path / "m")
    assert not dl.is_model_downloaded("fake")
    path = dl.download_model("fake", show_progress=False)
    assert dl.verify_model("fake") and dl.get_model_path("fake") == path
    assert dl.download_multiple_models(["fake", "silva"]) == {"fake": path, "silva": None}
    with pytest.raises(ValueError):
        dl.get_model_path("silva")
    # an old key finds a copy cached by 26.10.0 under its earlier name
    old = tmp_path / "m" / "gg2_v4_mrgroups_model_26.10.pt"
    old.write_bytes(b"x")
    assert dl.get_model_path("gg2_v4") == old and "gg2_v4" in dl.list_downloaded_models()
    assert md.superseded_model_warning is models.superseded_model_warning
