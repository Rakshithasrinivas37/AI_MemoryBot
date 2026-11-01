"""SSM orchestration tests without AWS credentials or network calls."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('deploy_ssm', Path(__file__).resolve().parents[1] / 'scripts/deploy_ssm.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
INSTANCE = 'i-0123456789abcdef0'
IMAGE = 'ghcr.io/example/memorybot@sha256:' + 'a' * 64
SENT = {'Command': {'CommandId': 'test-command'}}


class SSMDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.aws = patch.object(module, 'aws').start()
        self.sleep = patch.object(module.time, 'sleep').start()
        patch('builtins.print').start()
        self.addCleanup(patch.stopall)

    def test_payload_preserves_existing_rollback_script(self):
        payload = module.command_payload(INSTANCE, IMAGE)
        self.assertEqual(payload['InstanceIds'], [INSTANCE])
        self.assertEqual(payload['DocumentName'], 'AWS-RunShellScript')
        commands = payload['Parameters']['commands'][0]
        encoded = base64.b64encode(module.SCRIPT.read_bytes()).decode('ascii')
        self.assertIn(encoded, commands)
        self.assertIn(IMAGE, commands)
        self.assertIn('bash "$deploy_script"', commands)
        self.assertEqual(payload['Parameters']['executionTimeout'], ['900'])

    def test_eventual_consistency_and_pending_then_success(self):
        self.aws.side_effect = [SENT, subprocess.CalledProcessError(
            255, 'aws', stderr='InvocationDoesNotExist'), {'Status': 'InProgress'},
            {'Status': 'Success', 'ResponseCode': 0}]
        module.deploy(INSTANCE, IMAGE)
        self.assertEqual(self.sleep.call_count, 2)
        args = self.aws.call_args_list[0].args
        self.assertEqual(args[0], 'send-command')
        self.assertEqual(json.loads(args[2])['InstanceIds'], [INSTANCE])
        self.assertFalse(any(c.args[0] == 'cancel-command' for c in self.aws.call_args_list))

    def test_remote_failure_fails_job(self):
        self.aws.side_effect = [SENT, {'Status': 'Failed', 'ResponseCode': 1}]
        with self.assertRaisesRegex(RuntimeError, 'SSM deployment failed'):
            module.deploy(INSTANCE, IMAGE)
        self.assertEqual(self.aws.call_count, 2)

    def test_poll_timeout_requests_cancellation(self):
        self.aws.return_value = SENT
        with self.assertRaises(TimeoutError):
            module.deploy(INSTANCE, IMAGE, wait_seconds=0)
        self.assertEqual(self.aws.call_args.args[0], 'cancel-command')

    def test_permission_error_does_not_poll_forever(self):
        self.aws.side_effect = [SENT, subprocess.CalledProcessError(
            255, 'aws', stderr='AccessDeniedException'), {}]
        with self.assertRaises(subprocess.CalledProcessError):
            module.deploy(INSTANCE, IMAGE)
        self.sleep.assert_not_called()
        self.assertEqual(self.aws.call_args.args[0], 'cancel-command')

    def test_invalid_instance_or_image_never_sent(self):
        for instance, image in [('anything', IMAGE), (INSTANCE, 'ghcr.io/example/memorybot:latest')]:
            with self.subTest(instance=instance, image=image):
                with self.assertRaises(ValueError):
                    module.deploy(instance, image)
        self.aws.assert_not_called()

    def test_success_status_with_nonzero_code_still_fails(self):
        self.aws.side_effect = [SENT, {'Status': 'Success', 'ResponseCode': 1}]
        with self.assertRaises(RuntimeError):
            module.deploy(INSTANCE, IMAGE)


if __name__ == '__main__':
    unittest.main()
