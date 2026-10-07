"""Static gates for bounded FIRM 4B GPU automation."""
from pathlib import Path
import subprocess,unittest
ROOT=Path(__file__).resolve().parents[1]
class AutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch=(ROOT/'scripts/firm4b_gpu_launch.py').read_text()
        cls.startup=(ROOT/'scripts/firm4b_gpu_startup.sh').read_text()
        cls.watch=(ROOT/'scripts/firm4b_vps_watchdog.py').read_text()
    def test_bound_one_vm(self):
        self.assertIn('INSTANCE="firm-4b-train-01"',self.watch)
        self.assertIn('PROJECT="firm-gpu-experiments"',self.watch)
        self.assertIn('ZONE="us-east4-c"',self.watch)
        for arg in ('--max-run-duration=2h','--instance-termination-action=DELETE','--no-service-account','--no-scopes'):
            self.assertIn(arg,self.launch)
        self.assertIn('seconds!=7200',self.launch)
        self.assertIn('details.get("serviceAccounts")',self.launch)
        self.assertIn('d.get("autoDelete")',self.launch)
    def test_startup_pinned_and_timeout(self):
        self.assertEqual(self.startup.count('@FIRM_PINNED_COMMIT@'),2)
        self.assertIn('git checkout --detach "@FIRM_PINNED_COMMIT@"',self.startup)
        self.assertIn('--allow-provisional-data',self.startup)
        self.assertIn('timeout --signal=INT --kill-after=100s 5400s',self.startup)
        self.assertNotIn('gcloud auth',self.startup)
        subprocess.run(['bash','-n',str(ROOT/'scripts/firm4b_gpu_startup.sh')],check=True)
    def test_watchdog_recovery_and_cleanup(self):
        for name in ('snapshot()','delete_and_audit()','CREDENTIALS_REVOKED','REVOKE_FAILED'):
            self.assertIn(name,self.watch)
        for resource in ('instances','disks','addresses','reservations'):
            self.assertIn(f'"compute","{resource}","list"',self.watch)
        self.assertIn('call("auth","revoke","--all","--quiet"',self.watch)
    def test_launcher_dry_run_harmless(self):
        import sys
        p=subprocess.run([sys.executable,str(ROOT/'scripts/firm4b_gpu_launch.py')],
                         text=True,capture_output=True,cwd=ROOT)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('"max_run_duration_seconds": 7200',p.stdout)
        self.assertIn('authorization_required',p.stdout)
if __name__=='__main__':unittest.main()
