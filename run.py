import argparse
import json 
import yaml
import os 
import subprocess
import sys


def init():
    parser = argparse.ArgumentParser(description='Run a pipeline')
    parser.add_argument('--pipeline_path', type=str, metavar="PATH", required=True)
    parser.add_argument('--config_path', type=str, metavar="PATH", required=True)
    args = parser.parse_args()
    
    return args


def main():
    args = init()
    with open(args.config_path, 'r') as fp:
        config = yaml.safe_load(fp)
        config = config if config is not None else {}
        
    pipeline_path = os.path.dirname(os.path.abspath(args.pipeline_path))
    pipeline_module = os.path.splitext(os.path.basename(args.pipeline_path))[0]
    sys.path.append(pipeline_path)
    pipeline = getattr(__import__(pipeline_module), 'Pipeline')(**config)
    sys.path.pop()
    del sys.modules[pipeline_module]
    
    pipeline()
    
    
if __name__ == '__main__':
    main()