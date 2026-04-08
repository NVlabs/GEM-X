# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
import traceback
import uuid
from copy import deepcopy
from pathlib import Path
import sys

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo.demo_soma_fbx import run_demo_from_kwargs


class JobRequest(BaseModel):
    video: str
    output_root: str = "outputs/demo_soma"
    bvh_path: str | None = None
    render_incam: bool = False
    static_cam: bool = False
    verbose: bool = False
    render_mhr: bool = False
    render_scale: float = 0.5
    composite_background: bool = False
    ckpt: str = "inputs/pretrained/gem_soma.ckpt"
    exp: str = "gem_soma_regression"
    detector_name: str = "vitdet"
    sam3d_ckpt_path: str | None = None
    sam3d_mhr_path: str | None = None


JOB_LOCK = threading.Lock()
JOB_STORE: dict[str, dict] = {}


def _append_log(job_id, line):
    with JOB_LOCK:
        JOB_STORE[job_id]["logs"].append(line)


def _update_job(job_id, **fields):
    with JOB_LOCK:
        JOB_STORE[job_id].update(fields)


def _snapshot_job(job_id):
    with JOB_LOCK:
        job = JOB_STORE.get(job_id)
        return deepcopy(job) if job is not None else None


def _worker(job_id, params):
    def progress_callback(stage, message, percent=None):
        line = f"[{stage}] {message}"
        _append_log(job_id, line)
        _update_job(job_id, stage=stage, message=message, progress=percent or 0)

    try:
        _update_job(job_id, status="running")
        result = run_demo_from_kwargs(progress_callback=progress_callback, **params)
        _append_log(job_id, "[DONE] Outputs are ready.")
        _update_job(
            job_id,
            status="completed",
            progress=100,
            stage="DONE",
            message="Completed",
            result=result,
        )
    except Exception as exc:
        _append_log(job_id, f"[ERROR] {exc}")
        _append_log(job_id, traceback.format_exc())
        _update_job(
            job_id,
            status="failed",
            stage="ERROR",
            message=str(exc),
            error=str(exc),
        )


def create_job(request: JobRequest):
    params = request.model_dump()
    job_id = str(uuid.uuid4())
    with JOB_LOCK:
        JOB_STORE[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "QUEUED",
            "message": "Waiting to start",
            "progress": 0,
            "logs": ["[QUEUED] Job created."],
            "result": None,
            "error": None,
            "params": params,
        }

    thread = threading.Thread(target=_worker, args=(job_id, params), daemon=True)
    thread.start()
    return _snapshot_job(job_id)


app = FastAPI(title="GEM SOMA BVH Demo")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/jobs")
def create_job_endpoint(request: JobRequest):
    return create_job(request)


@app.get("/api/jobs/{job_id}")
def get_job_endpoint(job_id: str):
    job = _snapshot_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _start_job_ui(
    video,
    output_root,
    bvh_path,
    render_incam,
    static_cam,
    verbose,
    render_mhr,
    render_scale,
    composite_background,
    ckpt,
    exp,
    detector_name,
    sam3d_ckpt_path,
    sam3d_mhr_path,
):
    if not video:
        raise gr.Error("Video path or uploaded video is required.")

    request = JobRequest(
        video=str(video),
        output_root=output_root,
        bvh_path=bvh_path or None,
        render_incam=render_incam,
        static_cam=static_cam,
        verbose=verbose,
        render_mhr=render_mhr,
        render_scale=render_scale,
        composite_background=composite_background,
        ckpt=ckpt,
        exp=exp,
        detector_name=detector_name,
        sam3d_ckpt_path=sam3d_ckpt_path or None,
        sam3d_mhr_path=sam3d_mhr_path or None,
    )
    job = create_job(request)
    status = f"{job['status']} | {job['stage']} | {job['progress']}%"
    logs = "\n".join(job["logs"])
    return job["id"], status, job["progress"], logs, None, None, None


def _poll_job_ui(job_id):
    if not job_id:
        return "idle", 0, "", None, None, None

    job = _snapshot_job(job_id)
    if job is None:
        return "job not found", 0, "", None, None, None

    status = f"{job['status']} | {job['stage']} | {job['progress']}% | {job['message']}"
    logs = "\n".join(job["logs"])
    result = job.get("result") or {}
    video_path = result.get("video_path")
    bvh_path = result.get("bvh_path")
    details = {
        "job_id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "message": job["message"],
        "progress": job["progress"],
        "result": result,
        "error": job.get("error"),
    }

    if video_path and not Path(video_path).exists():
        video_path = None
    if bvh_path and not Path(bvh_path).exists():
        bvh_path = None

    return status, job["progress"], logs, video_path, bvh_path, details


with gr.Blocks(title="GEM SOMA BVH Demo") as demo:
    gr.Markdown(
        """
        # GEM SOMA BVH Demo
        3D body pose estimation을 실행하고 `In-cam` 렌더 영상과 `BVH` 파일을 생성합니다.
        """
    )

    job_id_state = gr.State("")

    with gr.Row():
        video = gr.File(label="Input Video", file_types=["video"])
        with gr.Column():
            video_path = gr.Textbox(label="Or Video Path", value="data/8turn_1_downsize_trim.mp4")
            output_root = gr.Textbox(label="Output Root", value="outputs/demo_soma")
            bvh_path = gr.Textbox(label="Custom BVH Path", value="")
            detector_name = gr.Textbox(label="Detector", value="vitdet")
            ckpt = gr.Textbox(label="Checkpoint", value="inputs/pretrained/gem_soma.ckpt")
            exp = gr.Textbox(label="Experiment", value="gem_soma_regression")

    with gr.Row():
        render_incam = gr.Checkbox(label="Render In-cam", value=False)
        static_cam = gr.Checkbox(label="Static Cam", value=False)
        verbose = gr.Checkbox(label="Verbose", value=False)
        render_mhr = gr.Checkbox(label="Render MHR", value=False)
        composite_background = gr.Checkbox(label="Composite Background", value=False)

    render_scale = gr.Slider(label="Render Scale", minimum=0.25, maximum=1.0, step=0.05, value=0.5)

    with gr.Accordion("Advanced", open=False):
        sam3d_ckpt_path = gr.Textbox(label="SAM3D Checkpoint", value="")
        sam3d_mhr_path = gr.Textbox(label="SAM3D MHR Path", value="")

    run_button = gr.Button("Run Pipeline", variant="primary")
    status_box = gr.Textbox(label="Status", interactive=False)
    progress_bar = gr.Slider(label="Progress", minimum=0, maximum=100, step=1, interactive=False)
    logs_box = gr.Textbox(label="Logs", lines=18, interactive=False)

    with gr.Row():
        output_video = gr.Video(label="In-cam Output")
        output_bvh = gr.File(label="BVH Output")

    details_json = gr.JSON(label="Job Details")
    poll_timer = gr.Timer(1.0)

    def _resolve_video_input(uploaded_video, typed_video_path):
        if uploaded_video is not None:
            candidate = getattr(uploaded_video, "name", None)
            if candidate:
                return candidate
            if isinstance(uploaded_video, str):
                return uploaded_video
        return typed_video_path

    run_button.click(
        fn=lambda uploaded_video, typed_video_path, output_root, bvh_path, render_incam, static_cam, verbose, render_mhr, render_scale, composite_background, ckpt, exp, detector_name, sam3d_ckpt_path, sam3d_mhr_path: _start_job_ui(
            _resolve_video_input(uploaded_video, typed_video_path),
            output_root,
            bvh_path,
            render_incam,
            static_cam,
            verbose,
            render_mhr,
            render_scale,
            composite_background,
            ckpt,
            exp,
            detector_name,
            sam3d_ckpt_path,
            sam3d_mhr_path,
        ),
        inputs=[
            video,
            video_path,
            output_root,
            bvh_path,
            render_incam,
            static_cam,
            verbose,
            render_mhr,
            render_scale,
            composite_background,
            ckpt,
            exp,
            detector_name,
            sam3d_ckpt_path,
            sam3d_mhr_path,
        ],
        outputs=[job_id_state, status_box, progress_bar, logs_box, output_video, output_bvh, details_json],
    )

    poll_timer.tick(
        fn=_poll_job_ui,
        inputs=[job_id_state],
        outputs=[status_box, progress_bar, logs_box, output_video, output_bvh, details_json],
    )


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(
        "scripts.demo.demo_soma_webapp:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        access_log=False,
        log_level="warning",
    )
