"""Standard-library tests run in .venv-vcc; every network operation is mocked."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import submit_public_auxiliary_vcc_v10 as upload
from vcc import api, auth, submit


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root/"prediction.vcc"
        self.package.write_bytes(b"synthetic-not-a-real-package")
        generation = self.root/"generation.json"
        upload.dump(generation,{"completed":True,"submission_performed":False})
        self.record = {"completed":True,"official_cli_version":"0.2.0","official_container_validated":True,
            "prep_result":{"n_cells":360000,"n_genes":18533,"normalization":"counts-preserved",
                "verified_targets":True,"dry_run":False,"cells_per_context":{"A":120000,"B":120000,"C":120000},"nnz":123},
            "package":{"path":str(self.package),"sha256":upload.safe.digest(self.package)},
            "generation_completion":{"path":str(generation),"sha256":upload.safe.digest(generation)}}
        self.completion = self.root/"package.json"
        upload.dump(self.completion,self.record)
        self.sha = upload.safe.digest(self.completion)
        self.out = self.root/"upload"
        self.env = patch.dict(os.environ,clear=False)
        self.env.start(); self.addCleanup(self.env.stop)

    def invoke(self,action="submit",seconds=0):
        with patch.object(upload.socket,"gethostname",return_value="cbsuvlaminck3"), \
             patch.object(upload.getpass,"getpass",return_value="vcc_pat_SYNTHETIC_SECRET"):
            return upload.execute(action,self.completion,self.sha,self.out,seconds)

    def successful_submit(self,**kwargs):
        self.assertEqual(kwargs["endpoint"],"https://virtualcellchallenge.org")
        self.assertFalse(kwargs["force"])
        self.assertFalse(kwargs["skip_limit_check"])
        self.assertTrue(kwargs["verify_targets"])
        self.assertTrue(kwargs["check_cell_counts"])
        self.assertFalse(kwargs["wait"])
        self.assertEqual(kwargs["token"],"vcc_pat_SYNTHETIC_SECRET")
        kwargs["on_event"]("created",entry_id="synthetic-entry")
        self.assertEqual(upload.safe.read_json(self.out/"entry.json")["entry_id"],"synthetic-entry")
        self.assertEqual((self.out/"entry.json").stat().st_mode & 0o777,0o600)
        kwargs["save_resume"]("synthetic-entry",{"entry_id":"synthetic-entry","synthetic":True})
        kwargs["on_event"]("upload_progress",done=1,total=2)
        kwargs["clear_resume"]("synthetic-entry")
        return submit.SubmitResult(entry_id="synthetic-entry",file_path="remote-file",model_name=upload.MODEL,
            is_final=False,bytes_uploaded=123,md5_verified=True)

    def test_success_persists_entry_before_upload_and_never_stores_token(self):
        with patch.object(submit,"run_submit",side_effect=self.successful_submit) as called, \
             patch.object(api,"get_submission",return_value={"entry_id":"synthetic-entry","status":"scoring","upload_url":"SECRET_URL"}):
            result = self.invoke()
        self.assertEqual(called.call_count,1)
        self.assertEqual(result,{"entry_id":"synthetic-entry","status":"scoring"})
        self.assertTrue((self.out/"submission.json").is_file())
        for path in self.out.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"vcc_pat_SYNTHETIC_SECRET",path.read_bytes())
                self.assertNotIn(b"SECRET_URL",path.read_bytes())

    def test_repeated_submit_refuses_existing_directory_before_auth_or_network(self):
        self.out.mkdir()
        with patch.object(upload.getpass,"getpass") as prompt,patch.object(submit,"run_submit") as call:
            with self.assertRaises(ValueError):
                upload.execute("submit",self.completion,self.sha,self.out,0)
        prompt.assert_not_called(); call.assert_not_called()

    def test_bad_monitor_refused_before_any_new_entry(self):
        for seconds in (-1,3601,True):
            with patch.object(submit,"run_submit") as call:
                with self.assertRaises(ValueError): self.invoke(seconds=seconds)
            call.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_wrong_hash_or_counts_validation_rejected(self):
        with self.assertRaises(ValueError): upload.checked_package(self.completion,"0"*64)
        for key,value in (("verified_targets",False),("dry_run",True),("normalization","normalized-log1p"),
                          ("nnz",4750000001),("n_cells",359600),("n_genes",482)):
            with self.subTest(key=key):
                copy = json.loads(json.dumps(self.record)); copy["prep_result"][key] = value
                path = self.root/(key+".json"); upload.dump(path,copy)
                with self.assertRaises(ValueError): upload.checked_package(path,upload.safe.digest(path))

    def test_package_bytes_changed_refused(self):
        self.package.write_bytes(b"changed")
        with self.assertRaises(ValueError): upload.checked_package(self.completion,self.sha)

    def test_missing_official_container_validation_refused(self):
        copy = dict(self.record); copy.pop("official_container_validated")
        path = self.root/"unvalidated.json"; upload.dump(path,copy)
        with self.assertRaises(ValueError): upload.checked_package(path,upload.safe.digest(path))

    def test_failed_upload_keeps_same_entry_for_resume(self):
        def fail(**kw):
            kw["on_event"]("created",entry_id="synthetic-entry")
            kw["save_resume"]("synthetic-entry",{"entry_id":"synthetic-entry","local_path":str(self.package),"model_name":upload.MODEL})
            raise submit.SubmitError("do-not-print-sensitive-url")
        with patch.object(submit,"run_submit",side_effect=fail):
            with self.assertRaises(SystemExit): self.invoke()
        self.assertEqual(upload.safe.read_json(self.out/"entry.json")["entry_id"],"synthetic-entry")
        self.assertFalse((self.out/"submission.json").exists())
        def resume(**kw):
            self.assertEqual(kw["resume"]["entry_id"],"synthetic-entry")
            self.assertIsNone(kw["pending_uploads"])
            return submit.SubmitResult("synthetic-entry","remote",upload.MODEL,False,123,True)
        with patch.object(submit,"run_submit",side_effect=resume) as call, \
             patch.object(api,"get_submission",side_effect=[{"entry_id":"synthetic-entry","status":"uploading"},
                 {"entry_id":"synthetic-entry","status":"scoring"}]):
            self.invoke("resume")
        self.assertEqual(call.call_count,1)

    def test_ambiguous_already_launched_resume_polls_without_relaunch(self):
        with patch.object(submit,"run_submit",side_effect=self.successful_submit), \
             patch.object(api,"get_submission",return_value={"entry_id":"synthetic-entry","status":"scoring"}):
            self.invoke()
        with patch.object(submit,"run_submit") as call, \
             patch.object(api,"get_submission",return_value={"entry_id":"synthetic-entry","status":"scoring"}):
            result = self.invoke("resume")
        call.assert_not_called(); self.assertEqual(result["status"],"scoring")

    def test_tampered_resume_package_refused(self):
        with patch.object(submit,"run_submit",side_effect=self.successful_submit), \
             patch.object(api,"get_submission",return_value={"entry_id":"synthetic-entry","status":"scoring"}):
            self.invoke()
        auth.save_pending_upload("default","synthetic-entry",{"entry_id":"synthetic-entry","model_name":upload.MODEL,
            "local_path":str(self.root/"another.vcc")})
        with patch.object(submit,"run_submit") as call, \
             patch.object(api,"get_submission",return_value={"entry_id":"synthetic-entry","status":"uploading"}):
            with self.assertRaises(ValueError): self.invoke("resume")
        call.assert_not_called()

    def test_public_status_excludes_sensitive_error_text(self):
        self.assertEqual(upload.public_status({"status":"failed","error_code":"bad_scale","error_message":"signedURL"}),
            {"status":"failed","error_code":"bad_scale"})

    def test_status_never_creates_new_entry(self):
        with patch.object(submit,"run_submit",side_effect=self.successful_submit), \
             patch.object(api,"get_submission",return_value={"entry_id":"synthetic-entry","status":"scoring"}):
            self.invoke()
        with patch.object(submit,"run_submit") as call, \
             patch.object(api,"get_submission",return_value={"entry_id":"synthetic-entry","status":"published","score_avg":-.03}):
            result = self.invoke("status")
        call.assert_not_called(); self.assertEqual(result["score_avg"],-.03)
        self.assertTrue((self.out/"status_00002.json").is_file())

    def test_private_dump_exclusive(self):
        path = self.root/"exclusive.json"; upload.dump(path,{"x":1})
        with self.assertRaises(FileExistsError): upload.dump(path,{"x":2})
        self.assertEqual(upload.safe.read_json(path),{"x":1})


if __name__ == "__main__": unittest.main()
