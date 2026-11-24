#!/usr/bin/env python3
"""Loop-depth generation probe.

Runs a frozen prompt set through llama.cpp at several loop counts (metadata override
`inkling2_loop.loop.count`) and records the assistant content + reasoning for each run.

Every model invocation goes through the Scion gpu_run.sh + gpu_guard.sh pair.

Example:
  python3 scripts/run_loop_prompts.py \
    --llama-cli  .../build-hip-loop/bin/llama-cli \
    --model      .../Inkling-2-Mini-UD-IQ4_XS.gguf \
    --gpu-run    .../scripts/scion/gpu_run.sh \
    --gpu-guard  .../scripts/scion/gpu_guard.sh \
    --loops 3,2,4 --out results/loop-gen.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_last_message(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    last = None
    for line in text.split("\n"):
        marker = "Parsed message: "
        i = line.find(marker)
        if i >= 0:
            last = line[i + len(marker):]
    if not last:
        return {}
    start = last.find("{")
    if start < 0:
        return {}
    # trim anything after the JSON object
    end = last.rfind("}")
    try:
        return json.loads(last[start:end + 1])
    except json.JSONDecodeError:
        return {}


def clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<\|[^|>]*\|>", "", text)
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-run", required=True)
    ap.add_argument("--gpu-guard", required=True)
    ap.add_argument("--prompts", default=str(ROOT / "eval" / "prompts.json"))
    ap.add_argument("--loops", default="3,2,4")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out", default=str(ROOT / "results" / "loop-gen.json"))
    ap.add_argument("--logs", default=str(ROOT / "logs"))
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    loops = [int(x) for x in args.loops.split(",") if x.strip()]
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)

    results = []
    for loop in loops:
        for p in prompts:
            log = logs / f"loopgen-{loop}-{p['id']}.log"
            cmd = [
                args.gpu_run, args.gpu_guard, args.llama_cli,
                "-m", args.model,
                "-p", p["prompt"],
                "--reasoning-effort", "none",
                "-n", str(args.n_predict),
                "--temp", "0",
                "-ngl", "99",
                "-c", str(args.ctx),
                "-v",
            ]
            if loop != 3:
                cmd += ["--override-kv", f"inkling2_loop.loop.count=int:{loop}"]
            env = dict(os.environ)
            env["GPU_RUN_LOG"] = str(log)
            env["GPU_RUN_PROGRESS_S"] = env.get("GPU_RUN_PROGRESS_S", "900")
            print(f"[loop {loop}] {p['id']}", flush=True)
            rc = subprocess.run(cmd, env=env).returncode
            msg = parse_last_message(log)
            content = clean(msg.get("content", ""))
            reasoning = clean(msg.get("reasoning_content", ""))
            exp = [e.lower() for e in p.get("expected", [])]
            low = content.lower()
            match = any(e in low for e in exp) if exp else None
            results.append({
                "loop": loop,
                "id": p["id"],
                "prompt": p["prompt"],
                "expected": p.get("expected", []),
                "content": content,
                "reasoning": reasoning,
                "match": match,
                "rc": rc,
                "log": str(log.relative_to(ROOT)) if str(log).startswith(str(ROOT)) else str(log),
            })
            print(f"    rc={rc} match={match} content={content[:100]!r}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
