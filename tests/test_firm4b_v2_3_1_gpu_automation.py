"""Static safety/integrity gates for bounded FIRM-4B v2.3.1 GPU automation."""
from pathlib import Path
import subprocess,unittest,sys
ROOT=Path(__file__).resolve().parents[1]

class V231GPUAutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch=(ROOT/'scripts/firm4b_v2_3_1_gpu_launch.py').read_text()
        cls.startup=(ROOT/'scripts/firm4b_v2_3_1_gpu_startup.sh').read_text()
        cls.watch=(ROOT/'scripts/firm4b_v2_3_1_vps_watchdog.py').read_text()

    def test_one_bounded_identity_free_vm(self):
        self.assertIn('INSTANCE="firm-4b-v2-3-1-train-01"',self.watch)
        self.assertIn('PROJECT="firm-gpu-experiments"',self.watch)
        self.assertIn('ZONE="northamerica-northeast2-a"',self.watch)
        for arg in ('--max-run-duration=2h','--instance-termination-action=DELETE',
                    '--no-service-account','--no-scopes'):
            self.assertIn(arg,self.launch)
        self.assertIn('seconds!=7200',self.launch)
        self.assertIn('details.get("serviceAccounts")',self.launch)
        self.assertIn('d.get("autoDelete")',self.launch)

    def test_startup_exact_commit_dataset_and_steps(self):
        self.assertEqual(self.startup.count('@FIRM_PINNED_COMMIT@'),2)
        self.assertIn('git checkout --detach "@FIRM_PINNED_COMMIT@"',self.startup)
        self.assertIn('train_firm4b_v2_3_1.py',self.startup)
        self.assertIn('data/processed/firm4b_v2_3_1_expanded_v1',self.startup)
        self.assertIn('--steps 1373',self.startup)
        self.assertIn('timeout --signal=INT --kill-after=100s 5400s',self.startup)
        self.assertIn('chmod -R a+rX "$FIRM_OUT"',self.startup)
        self.assertNotIn('gcloud auth',self.startup)
        subprocess.run(['bash','-n',str(ROOT/'scripts/firm4b_v2_3_1_gpu_startup.sh')],check=True)

    def test_watchdog_recovery_and_cleanup(self):
        self.assertIn('RUN="firm4b-v2-3-1-expanded-v1"',self.watch)
        self.assertIn('firm_v231_run.json',self.watch)
        self.assertIn('adapter_model.safetensors',self.watch)
        self.assertIn('chat_template.jinja',self.watch)
        for resource in ('instances','disks','addresses','reservations'):
            self.assertIn(f'"compute","{resource}","list"',self.watch)
        self.assertIn('call("auth","revoke","--all","--quiet"',self.watch)

    def test_launcher_dry_run_no_cloud(self):
        p=subprocess.run([sys.executable,str(ROOT/'scripts/firm4b_v2_3_1_gpu_launch.py')],
                         cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('"max_run_duration_seconds": 7200',p.stdout)
        self.assertIn('"stage": "authorization_required"',p.stdout)
        self.assertIn('"source_git_sha"',p.stdout)
        self.assertIn('"zone": "northamerica-northeast2-a"',p.stdout)

if __name__=='__main__':unittest.main()
