import json
from wisdomify.paths import CONF_JSON


def load_conf() -> dict:
    with open(CONF_JSON, 'r') as fh:
        return json.loads(fh.read())
