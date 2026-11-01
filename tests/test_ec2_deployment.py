"""Exercise the real deployment shell script with a fake Docker CLI; no AWS calls."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'deploy_ec2.sh'
IMAGE = 'ghcr.io/example/memorybot@sha256:' + 'a' * 64
FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
p = Path(os.environ['FAKE_STATE'])
s = json.loads(p.read_text())
a = sys.argv[1:]
s['calls'].append(a)
containers = s['containers']
result = 0
if a[0] == 'pull':
    result = 1 if s.get('fail_pull') else 0
elif a[:2] == ['container', 'inspect']:
    result = 0 if a[2] in containers else 1
elif a[0] == 'inspect':
    c = containers.get(a[-1])
    if c is None:
        result = 1
    elif 'Health.Status' in a[2]:
        print('unhealthy' if s.get('unhealthy') else 'healthy')
    else:
        print('true' if c['running'] else 'false')
elif a[0] == 'rename':
    containers[a[2]] = containers.pop(a[1])
elif a[0] == 'stop':
    containers[a[-1]]['running'] = False
elif a[0] == 'start':
    containers[a[1]]['running'] = True
elif a[0] == 'rm':
    containers.pop(a[-1], None)
elif a[0] == 'run':
    containers['memorybot'] = {'running': True, 'image': a[-1]}
    result = 1 if s.get('fail_run') else 0
p.write_text(json.dumps(s))
sys.exit(result)
'''


class DeploymentTests(unittest.TestCase):
    def run_deploy(self, *, old=True, image=IMAGE, **flags):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state.json'
            state.write_text(json.dumps({
                'calls': [], 'containers': {'memorybot': {'image': 'previous', 'running': True}} if old else {},
                **flags,
            }))
            for name, body in [('docker', FAKE_DOCKER), ('sleep', '#!/bin/sh\nexit 0\n'),
                               ('flock', '#!/bin/sh\nexit 0\n')]:
                executable = root / name
                executable.write_text(body)
                executable.chmod(0o755)
            # Keep the host's /var/lock untouched during these portable local tests.
            script = root / 'deploy.sh'
            script.write_text(SCRIPT.read_text().replace('/var/lock/memorybot-deploy.lock', str(root / 'deploy.lock')))
            result = subprocess.run(['bash', str(script), image], env={
                **os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'], 'FAKE_STATE': str(state),
            }, capture_output=True, text=True)
            return result, json.loads(state.read_text())

    def test_success_replaces_container_and_preserves_volume_mapping(self):
        result, state = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state['containers'], {'memorybot': {'image': IMAGE, 'running': True}})
        run = next(call for call in state['calls'] if call[0] == 'run')
        self.assertIn('memorybot-data:/data', run)
        self.assertIn('memorybot-models:/home/app/.cache/huggingface', run)
        self.assertIn('127.0.0.1:8000:8000', run)
        self.assertFalse(any(call[0] == 'volume' for call in state['calls']))

    def test_first_deployment(self):
        result, state = self.run_deploy(old=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state['containers']['memorybot']['image'], IMAGE)

    def test_failed_pull_does_not_stop_existing_container(self):
        result, state = self.run_deploy(fail_pull=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state['containers']['memorybot'], {'image': 'previous', 'running': True})
        self.assertFalse(any(c[0] in ('stop', 'rm', 'rename') for c in state['calls']))

    def test_unhealthy_image_restores_previous_running_container(self):
        result, state = self.run_deploy(unhealthy=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state['containers'], {'memorybot': {'image': 'previous', 'running': True}})

    def test_failed_run_restores_previous_container(self):
        result, state = self.run_deploy(fail_run=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state['containers'], {'memorybot': {'image': 'previous', 'running': True}})

    def test_failed_first_deployment_cleans_failed_container(self):
        result, state = self.run_deploy(old=False, unhealthy=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state['containers'], {})

    def test_unpinned_image_is_rejected(self):
        result, state = self.run_deploy(image='ghcr.io/example/memorybot:latest')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state['calls'], [])


if __name__ == '__main__':
    unittest.main()
