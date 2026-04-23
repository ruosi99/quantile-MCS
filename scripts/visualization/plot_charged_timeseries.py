import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_station_ids(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts


def subset_by_time_length(df: pd.DataFrame, time_length: Optional[str]) -> pd.DataFrame:
    if time_length is None:
        return df
    # 若是纯数字，按步数截取（前N行）
    tl = str(time_length).strip()
    if tl.isdigit():
        n = int(tl)
        if n <= 0:
            return df.iloc[0:0]
        return df.iloc[: min(n, len(df))]
    # 尝试解析为时间跨度，如 '24h', '7D', '30min'
    try:
        delta = pd.to_timedelta(tl)
        # 仅当索引为 datetime 可用
        if isinstance(df.index, pd.DatetimeIndex):
            if len(df.index) == 0:
                return df
            end_time = df.index[0] + delta
            return df.loc[df.index <= end_time]
    except Exception:
        pass
    # 解析失败则原样返回
    return df


def resolve_data_path(metric: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    data_dir = os.path.join(project_root, "data", "datasets", "charged", "SZH")
    filename = "volume.csv" if metric == "volume" else "duration.csv"
    return os.path.join(data_dir, filename)


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    # 行索引为时间，尝试解析为 datetime；列为充电站ID，统一转为字符串
    try:
        dt_index = pd.to_datetime(df.index, errors="coerce")
        if dt_index.notna().all():
            df.index = dt_index
            # 确保时间从早到晚排序
            df = df.sort_index()
    except Exception:
        pass
    df.columns = df.columns.astype(str)
    return df


def plot_stations(
    df: pd.DataFrame,
    station_ids: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = True,
):
    # 行索引为时间戳，若不是 datetime 则尝试转为 datetime
    try:
        timestamps = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        timestamps = pd.Index(df.index)

    plt.figure(figsize=(12, 5))
    plt.rcParams["axes.unicode_minus"] = False

    available_ids = set(df.columns)
    missing = [sid for sid in station_ids if sid not in available_ids]
    if missing:
        print(f"警告：以下站点ID未在数据中找到，将被忽略：{missing}")

    plotted_any = False
    for sid in station_ids:
        if sid not in available_ids:
            continue
        series = df[sid]
        # 确保按时间顺序绘制
        series = series.reindex(df.index)
        plt.plot(timestamps, pd.to_numeric(series, errors="coerce"), label=f"station {sid}")
        plotted_any = True

    if not plotted_any:
        raise ValueError("没有可绘制的站点。请检查输入的站点ID。")

    plt.xlabel("时间")
    plt.ylabel("数值")
    if title:
        plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        print(f"已保存图像到：{save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "可视化充电站用电需求数据。数据文件为矩阵：行=时间，列=充电站ID。"
        )
    )
    parser.add_argument(
        "--metric",
        choices=["volume", "duration"],
        default="volume",
        help="选择可视化的数据类型：volume 或 duration，默认 volume",
    )
    parser.add_argument(
        "--station-ids",
        type=parse_station_ids,
        required=True,
        help="需要查看的充电站ID，逗号分隔，例如: 1001,1002",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="可选，保存图片的路径（包含文件名）。不传则不保存",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="保存图片的清晰度DPI，默认150",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不显示图像，仅用于批量保存",
    )
    parser.add_argument(
        "--time-length",
        type=str,
        default=None,
        help=(
            "从起始时间起的可视化长度：整数步数（如 100）或时间跨度（如 24h, 7D, 30min）"
        ),
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    csv_path = resolve_data_path(args.metric)
    df = load_dataset(csv_path)
    # 从最开始时间起，按给定长度截取
    df = subset_by_time_length(df, getattr(args, "time_length", None))

    # 默认保存路径
    save_path = args.save
    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
        default_dir = os.path.join(project_root, "data", "results", "visualization")
        metric_tag = args.metric
        id_tag = "_".join(args.station_ids)
        save_path = os.path.join(default_dir, f"{metric_tag}_stations_{id_tag}.png")

    title = f"{args.metric} - stations: {', '.join(args.station_ids)}"
    plot_stations(
        df=df,
        station_ids=args.station_ids,
        title=title,
        save_path=save_path if args.no_show or args.save else None,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()


