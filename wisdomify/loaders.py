import json
from wisdomify.paths import CONF_JSON


def load_conf() -> dict:
    with open(CONF_JSON, 'r', encoding="utf-8") as fh:
        return json.loads(fh.read())
