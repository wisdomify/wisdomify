from wisdomify.loaders import load_conf

CONFIGS = load_conf()
VERSIONS = CONFIGS['versions']

RUN_VER = CONFIGS['run_ver']
RUN_PARAMS = CONFIGS[str(RUN_VER)]
