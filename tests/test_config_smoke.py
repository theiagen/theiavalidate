from theiavalidate import Config, TypeSpec

def test_typespec():
    assert str(TypeSpec.parse("set[float]")) == "set[float]"

def test_config_smoke():
    cfg = Config.from_dict({"key": "s", "columns": {"c": {"method": "exact", "type": "str"}}})
    assert cfg.key == "s"
