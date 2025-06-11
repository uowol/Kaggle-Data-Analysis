import argparse
import yaml
import os
import subprocess
import sys
from pathlib import Path


def init():
    parser = argparse.ArgumentParser(description='Run a project')
    parser.add_argument('--project_name', type=str, metavar="NAME", required=True, help='Name of the project to run')
    parser.add_argument('--pipeline_name', type=str, metavar="NAME", default="default", help='Name of the pipeline to run (default: default)')
    args = parser.parse_args()
    
    return args


def main():
    args = init()
    
    base_dir = Path(__file__).resolve().parent / args.project_name
    pipeline_path = f"{base_dir}/src/pipelines/{args.pipeline_name}/pipeline.py"
    config_path = f"{base_dir}/src/pipelines/{args.pipeline_name}/pipeline.yaml"
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
        config = config if config is not None else {}

    # -- install dependencies
    requirements_path = f"{args.project_name}/requirements.txt"
    if os.path.exists(requirements_path):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])

    # -- run pipeline module
    pipeline_dir = os.path.dirname(os.path.abspath(pipeline_path))
    pipeline_module = os.path.splitext(os.path.basename(pipeline_path))[0]
    sys.path.append(pipeline_dir)
    pipeline = getattr(__import__(pipeline_module), 'Pipeline')(**config)
    sys.path.pop()
    del sys.modules[pipeline_module]
    pipeline()
    
    
if __name__ == '__main__':
    main()