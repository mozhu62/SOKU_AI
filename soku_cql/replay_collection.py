from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CollectionJob:
    index: int
    total: int
    replay: Path
    relative: Path
    output: Path


@dataclass(frozen=True)
class CollectionResult:
    job: CollectionJob
    exit_code: int
    complete: bool
    error: str = ""


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _output_paths(output: Path) -> tuple[Path, Path]:
    return output.with_suffix(".objects.csv"), Path(f"{output}.done")


def _is_complete(output: Path) -> bool:
    objects, done = _output_paths(output)
    if not (output.is_file() and objects.is_file() and done.is_file()):
        return False
    try:
        # 旧三件套没有技能列，必须自动重采；无需用户逐个删除 CSV 或强制覆盖。
        with output.open("rb") as stream:
            header = stream.read(256).decode("utf-8-sig", errors="replace")
        return "# soku_training_csv_version=4\n" in header.replace("\r\n", "\n")
    except OSError:
        return False


def _run_game(job: CollectionJob, game: Path, speed: int) -> CollectionResult:
    environment = os.environ.copy()
    # REP与输出路径必须放进各自子进程的环境，不能修改父进程的全局环境。
    environment["SOKU_DATA_SOURCE_REPLAY"] = str(job.replay)
    environment["SOKU_DATA_OUTPUT_FILE"] = str(job.output)
    environment["SOKU_DATA_FAST_FORWARD"] = str(speed)
    try:
        process = subprocess.run(
            [str(game), str(job.replay)],
            cwd=game.parent,
            env=environment,
            check=False,
        )
        return CollectionResult(job, process.returncode, _is_complete(job.output))
    except OSError as error:
        return CollectionResult(job, -1, False, str(error))


def collect_replays(
    config: dict,
    overwrite: bool = False,
    workers_override: int | None = None,
) -> Path:
    """并行启动多个一次性游戏进程，分别采集对应的Replay。"""
    settings = config.get("collection")
    if not isinstance(settings, dict):
        raise ValueError("配置缺少 collection，无法启动游戏采集 REP")

    game = _resolve(settings["game_executable"])
    replay_directory = _resolve(settings["replay_dir"])
    capture_directory = _resolve(settings["capture_dir"])
    speed = int(settings.get("speed", 8))
    workers = int(workers_override if workers_override is not None else settings.get("workers", 2))

    if not game.is_file():
        raise FileNotFoundError(f"游戏程序不存在：{game}")
    if not replay_directory.is_dir():
        raise FileNotFoundError(f"REP 目录不存在：{replay_directory}")
    if not 1 <= speed <= 64:
        raise ValueError("collection.speed 必须在 1～64 之间")
    if not 1 <= workers <= 16:
        raise ValueError("collection.workers 必须在 1～16 之间")

    replays = sorted(
        path for path in replay_directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".rep"
    )
    if not replays:
        raise FileNotFoundError(f"REP 目录中没有 .rep 文件：{replay_directory}")

    capture_directory.mkdir(parents=True, exist_ok=True)
    jobs: list[CollectionJob] = []
    skipped = 0
    for index, replay in enumerate(replays, 1):
        relative = replay.relative_to(replay_directory)
        output = (capture_directory / relative).with_suffix(".csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and _is_complete(output):
            skipped += 1
            continue
        jobs.append(CollectionJob(index, len(replays), replay, relative, output))

    print(
        f"发现 {len(replays)} 个 REP：待采集 {len(jobs)}，跳过 {skipped}，"
        f"并行实例 {workers}，速度 {speed}"
    )
    if not jobs:
        return capture_directory

    succeeded = 0
    failed: list[CollectionResult] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="soku-replay") as executor:
        futures = {executor.submit(_run_game, job, game, speed): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            # 游戏退出码可能受其他模组影响，训练数据是否成功只以三件套为准。
            if result.complete:
                succeeded += 1
                suffix = "" if result.exit_code == 0 else f"，忽略退出码 {result.exit_code}"
                print(f"[{result.job.index}/{result.job.total}] OK: {result.job.relative}{suffix}")
            else:
                failed.append(result)
                detail = f"；{result.error}" if result.error else ""
                print(
                    f"[{result.job.index}/{result.job.total}] FAILED: {result.job.relative} "
                    f"exit={result.exit_code} complete=0{detail}"
                )

    print(f"采集结束：成功 {succeeded}，跳过 {skipped}，失败 {len(failed)}")
    if failed:
        names = "、".join(str(item.job.relative) for item in failed[:5])
        suffix = "……" if len(failed) > 5 else ""
        raise RuntimeError(f"有 {len(failed)} 个 REP 没有生成完整三件套：{names}{suffix}")
    return capture_directory
