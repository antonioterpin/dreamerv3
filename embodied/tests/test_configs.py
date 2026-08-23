import pathlib

import elements
import ruamel.yaml as yaml

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / 'dreamerv3' / 'configs.yaml'


def test_run_args_declare_the_resume_regex():
  configs = yaml.YAML(typ='safe').load(CONFIGS.read_text())
  config = elements.Config(configs['defaults'])
  args = elements.Config(**config.run, logdir=config.logdir)
  assert args.from_checkpoint == ''
  assert args.from_checkpoint_regex == '', (
      'parallel/train pass it to agent.load; empty loads every parameter')
