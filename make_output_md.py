import os
import pickle
import numpy as np

# stats 파일에 display명으로 저장 -> total_execution이 저장되어 있음.
# data['Class_name']['total_execution']으로 접근
# 접근하면, mean, std, min, max, count 값이 key로 해서 value들이 저장되어 있음.

# stats 파일에 클래스명으로 저장 -> optimization time이 저장되어 있음.
# data['Class_name']['optimization']으로 접근
# 접근하면, mean, std, min, max, count 값이 key로 해서 value들이 저장되어 있음.

TOTAL_EXECUTION_LIST = [
                     'BiRoLF (Ours)',
                     'BiRoLF w/o Blockwise (Ours)',
                     'RoLF'
                     ]

OPTIMIZATION_LIST = [
                     'BiRoLFLasso_Blockwise',
                     'BiRoLFLasso',
                     'RoLFLasso',
]

KEY2KEY_DICT = {
                'BiRoLF (Ours)': 'BiRoLFLasso_Blockwise',
                'BiRoLF w/o Blockwise (Ours)': 'BiRoLFLasso',
                'RoLF':'RoLFLasso'
                }


def read_regret(regret_path: str) -> dict:
    """
    Returns: {agent_name: {'T25': val, 'T50': val, 'T75': val, 'T100': val}}
    where val is the mean cumulative regret across trials at the given horizon fraction.
    """
    with open(regret_path, "rb") as f:
        data = pickle.load(f)
        _, regret_results = data

    result = {}
    for agent_name, arr in regret_results.items():
        arr = np.stack([np.asarray(a, dtype=float) for a in arr])  # (trials, horizon)
        T = arr.shape[1]
        
        mean_regret = arr.mean(axis=0)  # (horizon,)
        std_regret = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(T)

        indices = {
            'T25':  int(T * 0.25) - 1,
            'T50':  int(T * 0.50) - 1,
            'T75':  int(T * 0.75) - 1,
            'T100': T - 1,
        }
        result[agent_name] = {
            key: (float(mean_regret[idx]), float(std_regret[idx]))
            for key, idx in indices.items()
        }
    return result


def read_computation_timing(timing_path: str) -> dict:
    """
    Returns: {display_name: {'total_execution': {'mean': ..., 'std': ...},
                              'optimization':    {'mean': ..., 'std': ...}}}
    for each of the 3 models in TOTAL_EXECUTION_LIST (display names).
    """
    with open(timing_path, "rb") as f:
        data = pickle.load(f)

    result = {}
    for display_name in TOTAL_EXECUTION_LIST:  # display names
        class_name = KEY2KEY_DICT[display_name]  # class names

        te = data[display_name]['total_execution']
        total_exec = {'mean': te.get('mean'), 'std': te.get('std')}

        op = data[class_name]['optimization']
        optim = {'mean': op.get('mean'), 'std': op.get('std')}

        result[display_name] = {
            'total_execution': total_exec,
            'optimization': optim,
        }
    return result


def _fmt(val, digits=4):
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


def make_md(file_paths: list, output_path: str):
    """
    file_paths: alternating [regret_path_1, timing_path_1, regret_path_2, timing_path_2, ...]
    Saves one .md file per experiment pair to output_path/실험 {k}/{stem}.md
    """
    assert len(file_paths) % 2 == 0, "file_paths must have even length (regret, timing pairs)"

    pairs = [(file_paths[i], file_paths[i + 1]) for i in range(0, len(file_paths), 2)]

    for k, (regret_path, timing_path) in enumerate(pairs, start=1):
        regret_data = read_regret(regret_path)
        timing_data = read_computation_timing(timing_path)

        stem = os.path.splitext(os.path.basename(regret_path))[0]
        save_dir = os.path.join(output_path, f"실험 {k}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{stem}.md")

        lines = []
        lines.append(f"# 실험 {k}: {stem}\n")

        # ── Cumulative Regret Table ──
        lines.append("## Cumulative Regret (mean ± std)\n")
        lines.append("| Agent | T×0.25 | T×0.50 | T×0.75 | T×1.00 |")
        lines.append("|-------|--------|--------|--------|--------|")
        for agent_name, checkpoints in regret_data.items():
            row_vals = []
            for key in ['T25', 'T50', 'T75', 'T100']:
                mean, std = checkpoints[key]
                row_vals.append(f"{_fmt(mean)} ± {_fmt(std)}")
            lines.append(f"| {agent_name} | {' | '.join(row_vals)} |")
        lines.append("")

        # ── Optimization Time Table ──
        lines.append("## Optimization Time per Step (mean ± std, seconds)\n")
        lines.append("| Agent | Mean | Std |")
        lines.append("|-------|------|-----|")
        for display_name in TOTAL_EXECUTION_LIST:
            op = timing_data[display_name]['optimization']
            lines.append(
                f"| {display_name} | {_fmt(op.get('mean'))} | {_fmt(op.get('std'))} |"
            )
        lines.append("")

        # ── Total Execution Time Table ──
        lines.append("## Total Execution Time (mean ± std, seconds)\n")
        lines.append("| Agent | Mean | Std |")
        lines.append("|-------|------|-----|")
        for display_name in TOTAL_EXECUTION_LIST:
            te = timing_data[display_name]['total_execution']
            lines.append(
                f"| {display_name} | {_fmt(te.get('mean'))} | {_fmt(te.get('std'))} |"
            )
        lines.append("")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Saved → {save_path}")


if __name__ == "__main__":
    base_path = os.getcwd() + "/4. Rebuttal"
    
    file_paths = []
    
    while True:
        now_file = input("Regret File: ")
        if now_file in ["STOP","S","","Done"]:
            break
        file_paths.append(base_path+"/"+now_file)
        now_file = input("Timing File: ")
        file_paths.append(base_path+"/"+now_file)

    make_md(
        file_paths=file_paths,
        output_path=base_path + "/1. MD output",
    )
