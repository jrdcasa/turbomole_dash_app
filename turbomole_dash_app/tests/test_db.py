"""SQLite job-store tests."""

from backend.db import init_db, insert_job, update_job, get_job, list_jobs, JobRecord


def test_db_roundtrip(tmp_path):
    init_db(tmp_path / "jobs.sqlite")
    job = JobRecord(
        name="test", cluster="example", state="DRAFT",
        task_type="single_point",
        local_dir=str(tmp_path / "local"),
        remote_dir="/scratch/me/test",
        functional="BP86", basis_set="def2-SVP",
        submit_meta={"partition": "cpu", "nodes": 1},
    )
    jid = insert_job(job)
    assert jid > 0

    fetched = get_job(jid)
    assert fetched["name"] == "test"
    assert fetched["submit_meta"]["partition"] == "cpu"
    assert fetched["state"] == "DRAFT"

    update_job(jid, state="SUBMITTED", slurm_id="123456")
    fetched = get_job(jid)
    assert fetched["state"] == "SUBMITTED"
    assert fetched["slurm_id"] == "123456"

    all_jobs = list_jobs()
    assert len(all_jobs) == 1
