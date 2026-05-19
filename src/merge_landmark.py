import pandas as pd
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]

    main_all = base_dir / "data" / "landmarks" / "all.csv"
    front_all = base_dir / "data" / "landmarks" / "sitting_posture" / "front" / "all.csv"
    out_path = base_dir / "data" / "landmarks" / "all_merged.csv"

    if not main_all.is_file():
        raise FileNotFoundError(main_all)
    if not front_all.is_file():
        raise FileNotFoundError(front_all)

    print(f"Loading: {main_all}")
    df_main = pd.read_csv(main_all)

    print(f"Loading: {front_all}")
    df_front = pd.read_csv(front_all)

    df_main["source"] = "main_all"
    df_front["source"] = "sitting_front"

    merged = pd.concat([df_main, df_front], ignore_index=True)

    print("Merged shape:", merged.shape)
    print(f"Saving merged dataset to: {out_path}")
    merged.to_csv(out_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()