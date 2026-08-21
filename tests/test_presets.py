"""Preset resolution: local --preset-dir, and cached download from the repo."""

import urllib.error

import pytest

from theiavalidate import presets


class TestLocalPresetDir:
    def test_reads_named_yaml_from_dir(self, tmp_path):
        (tmp_path / "theiaprok_pe.yaml").write_text("key: id\n")
        path = presets.resolve_preset("theiaprok_pe", preset_dir=tmp_path)
        assert path == tmp_path / "theiaprok_pe.yaml"

    def test_accepts_name_with_yaml_suffix(self, tmp_path):
        (tmp_path / "theiaprok_pe.yaml").write_text("key: id\n")
        path = presets.resolve_preset("theiaprok_pe.yaml", preset_dir=tmp_path)
        assert path == tmp_path / "theiaprok_pe.yaml"

    def test_missing_preset_errors(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            presets.resolve_preset("nope", preset_dir=tmp_path)

    def test_no_network_used_for_local_dir(self, tmp_path, monkeypatch):
        (tmp_path / "p.yaml").write_text("key: id\n")

        def _boom(*a, **k):
            raise AssertionError("download must not be called for --preset-dir")

        monkeypatch.setattr(presets.urllib.request, "urlopen", _boom)
        presets.resolve_preset("p", preset_dir=tmp_path)


class TestRemoteUrl:
    def test_default_ref(self):
        url = presets.remote_url("theiaprok_pe")
        assert url == (
            "https://raw.githubusercontent.com/theiagen/"
            "public_health_bioinformatics/main/tests/config/theiavalidate/"
            "theiaprok_pe.yaml"
        )

    def test_custom_ref_in_url(self):
        assert "/v2.0.0/" in presets.remote_url("theiaprok_pe", ref="v2.0.0")


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestDownloadAndCache:
    def _isolate_cache(self, tmp_path, monkeypatch):
        # point XDG_CACHE_HOME at a temp dir so we never touch the real cache
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def test_downloads_and_caches(self, tmp_path, monkeypatch):
        self._isolate_cache(tmp_path, monkeypatch)
        calls = []

        def fake_urlopen(url):
            calls.append(url)
            return _FakeResponse(b"key: id\ncolumns: {}\n")

        monkeypatch.setattr(presets.urllib.request, "urlopen", fake_urlopen)

        first = presets.resolve_preset("theiaprok_pe")
        assert first.read_text().startswith("key: id")
        assert len(calls) == 1

        # second call is served from cache, no new download
        second = presets.resolve_preset("theiaprok_pe")
        assert second == first
        assert len(calls) == 1

    def test_refresh_forces_redownload(self, tmp_path, monkeypatch):
        self._isolate_cache(tmp_path, monkeypatch)
        calls = []

        def fake_urlopen(url):
            calls.append(url)
            return _FakeResponse(b"key: id\n")

        monkeypatch.setattr(presets.urllib.request, "urlopen", fake_urlopen)
        presets.resolve_preset("p")
        presets.resolve_preset("p", refresh=True)
        assert len(calls) == 2

    def test_404_gives_unknown_preset_error(self, tmp_path, monkeypatch):
        self._isolate_cache(tmp_path, monkeypatch)

        def fake_urlopen(url):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        monkeypatch.setattr(presets.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(ValueError, match="unknown preset 'bogus'"):
            presets.resolve_preset("bogus")

    def test_network_error_is_wrapped(self, tmp_path, monkeypatch):
        self._isolate_cache(tmp_path, monkeypatch)

        def fake_urlopen(url):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(presets.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(ValueError, match="could not reach"):
            presets.resolve_preset("theiaprok_pe")
