"""Static safety/integrity gates for bounded FIRM-4B v2.1 GPU automation."""
from pathlib import Path
import subprocess,unittest
ROOT=Path(__file__).resolve().parents[1]

class V21GPUAutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch=(ROOT/'scripts/firm4b_v2_1_gpu_launch.py').read_text()
        cls.startup=(ROOT/'scripts/firm4b_v2_1_gpu_startup.sh').read_text()
        cls.watch=(ROOT/'scripts/firm4b_v2_1_vps_watchdog.py').read_text()

    def test_one_bounded_identity_free_vm(self):
        self.assertIn('INSTANCE="firm-4b-v2-1-train-01"',self.watch)
        self.assertIn('PROJECT="firm-gpu-experiments"',self.watch)
        self.assertIn('ZONE="us-east4-a"',self.watch)
        for arg in ('--max-run-duration=2h','--instance-termination-action=DELETE',
                    '--no-service-account','--no-scopes'):
            self.assertIn(arg,self.launch)
        self.assertIn('seconds!=7200',self.launch)
        self.assertIn('details.get("serviceAccounts")',self.launch)
        self.assertIn('d.get("autoDelete")',self.launch)

    def test_startup_exact_commit_dataset_and_steps(self):
        self.assertEqual(self.startup.count('@FIRM_PINNED_COMMIT@'),2)
        self.assertIn('git checkout --detach "@FIRM_PINNED_COMMIT@"',self.startup)
        self.assertIn('train_firm4b_v2_1.py',self.startup)
        self.assertIn('data/processed/firm4b_v2_foundation_v2',self.startup)
        self.assertIn('--steps 1250',self.startup)
        self.assertIn('timeout --signal=INT --kill-after=100s 5400s',self.startup)
        self.assertIn('chmod -R a+rX "$FIRM_OUT"',self.startup)
        self.assertIn('PERM_HELPER_PID',self.startup)
        self.assertNotIn('gcloud auth',self.startup)
        subprocess.run(['bash','-n',str(ROOT/'scripts/firm4b_v2_1_gpu_startup.sh')],check=True)

    def test_watchdog_exact_run_and_cleanup(self):
        self.assertIn('RUN="firm4b-v2-1-expanded-v1"',self.watch)
        self.assertIn('firm_v2_run.json',self.watch)
        self.assertIn('adapter_model.safetensors',self.watch)
        for resource in ('instances','disks','addresses','reservations'):
            self.assertIn(f'"compute","{resource}","list"',self.watch)
        self.assertIn('call("auth","revoke","--all","--quiet"',self.watch)

    def test_launcher_dry_run_no_cloud(self):
        import sys
        p=subprocess.run([sys.executable,str(ROOT/'scripts/firm4b_v2_1_gpu_launch.py')],
                         cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('"max_run_duration_seconds": 7200',p.stdout)
        self.assertIn('"stage": "authorization_required"',p.stdout)
        self.assertIn('"source_git_sha"',p.stdout)

if __name__=='__main__':unittest.main()
