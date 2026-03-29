import shutil
from pathlib import Path


BASE_DIR = Path.cwd().parent
  
 
def organise_dataset(src: str = '/dataset' , dst: str = '/data/raw'):
    """
    Reads YOLO-labelled images from:
        {src}/{split}/images/  +  {src}/{split}/labels/
 
    And copies images (no labels) into:
        {dst}/{split}/correct/   ← class_id 0
        {dst}/{split}/incorrect/ ← class_id 1
 
    Args:
        src: Root folder containing train/test/valid (default: "dataset")
        dst: Destination root (default: "data/raw")
    """
    
    src_root = Path(str(BASE_DIR) + src)
    dst_root = Path(str(BASE_DIR) + dst)

    splits     = ["train", "test", "valid"]
    class_map  = {0: "correct", 1: "incorrect"}
 
    copied = {split: {cls: 0 for cls in class_map.values()} for split in splits}
    skipped = []
 
    for split in splits:
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
 
        if not img_dir.exists():
            print(f"[SKIP] {img_dir} not found — skipping {split}")
            continue
 
        # Create destination class folders
        for cls_name in class_map.values():
            (dst_root / split / cls_name).mkdir(parents=True, exist_ok=True)
 
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file():
                continue
 
            lbl_path = lbl_dir / (img_path.stem + ".txt")
 
            if not lbl_path.exists():
                print(f"[WARN] No label for {img_path.name} — skipped")
                skipped.append(str(img_path))
                continue
 
            
            # Read class_id from first line of label file
            content = lbl_path.read_text().strip()
            if not content:
                print(f"[WARN] Empty label file for {img_path.name} — skipped")
                skipped.append(str(img_path))
                continue
            
            first_line = content.splitlines()[0]
            class_id   = int(first_line.split()[0])
 
            if class_id not in class_map:
                print(f"[WARN] Unknown class_id {class_id} in {lbl_path.name} — skipped")
                skipped.append(str(img_path))
                continue
 
            cls_name = class_map[class_id]
            dst_path = dst_root / split / cls_name / img_path.name
 
            shutil.copy2(img_path, dst_path)
            copied[split][cls_name] += 1
 
    # Summary
    print("\n── Organised dataset summary ──────────────────────")
    total = 0
    for split in splits:
        for cls_name, n in copied[split].items():
            print(f"  {split:>5} / {cls_name:<10}  →  {n} images")
            total += n
    print(f"  {'TOTAL':<17}  →  {total} images")
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped (see warnings above)")
    else:
        print("\nNo files skipped")
    print(f"\n  Output: {dst_root.resolve()}")
 
 
if __name__ == "__main__":
    organise_dataset()