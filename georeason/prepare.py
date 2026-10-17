"""Build training and evaluation manifests from released annotations."""
import argparse
import json
import pickle
import re
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def earthreason(root, split):
    base = root / "EarthReason" / split
    rows = []
    for path in sorted((base / "QAs").glob("*.json")):
        data = read_json(path)
        image = base / "images" / (path.stem + ".jpg")
        if not image.exists():
            image = image.with_suffix(".png")
        rows.append(dict(id=path.stem, task="reasoning", image=str(image.relative_to(root)),
                         mask=str((base / "labels" / (path.stem + ".png")).relative_to(root)),
                         questions=data["questions"], reasoning=data.get("answer", [])))
    return rows


def rrsisd(root):
    folder = root / "RRSIS-D" / "rrsisd"
    with (folder / "refs(unc).p").open("rb") as f:
        refs = pickle.load(f)
    annotations = read_json(folder / "instances.json")
    images = {x["id"]: x for x in annotations["images"]}
    masks = {x["id"]: x for x in annotations["annotations"]}
    splits = {s: [] for s in ["train", "val", "test"]}
    held_out = set()
    for ref in refs:
        image = images[ref["image_id"]]
        row = dict(id=str(ref["ref_id"]), task="referring",
                   image="RRSIS-D/JPEGImages/" + image["file_name"],
                   width=image["width"], height=image["height"],
                   segmentation=masks[ref["ann_id"]]["segmentation"],
                   questions=[s["sent"] for s in ref["sentences"]])
        splits[ref["split"]].append(row)
        if ref["split"] != "train":
            held_out.add(int(Path(image["file_name"]).stem))
    return splits, held_out


def auxiliary(rows, held_out, kind, image_dir):
    pattern = r"\d{5}\.png" if kind == "geochat" else r"\d{5}_\d{4}\.png"
    excluded = r"\[(refer|identify|grounding)\]" if kind == "geochat" else r"\[refer\]"
    result = []
    for row in rows:
        name = row["image"]
        if not re.fullmatch(pattern, name) or int(name[:5]) in held_out:
            continue
        turns = row["conversations"]
        if any(re.search(excluded, t["value"], re.I) for t in turns if t["from"] == "human"):
            continue
        turns = [dict(t, value=re.sub(r"\[(vqa|caption)\]\s*", "", t["value"], flags=re.I))
                 if t["from"] == "human" else t for t in turns]
        result.append(dict(id=str(row["id"]), task="qa", image=str(Path(image_dir) / name),
                           conversations=turns))
    return result


def build(root, geochat_images, vrsbench_images):
    splits, held_out = rrsisd(root)
    data = {f"rrsisd_{split}": rows for split, rows in splits.items()}
    data.update({f"earthreason_{s}": earthreason(root, s) for s in splits})
    data["geochat_train"] = auxiliary(read_json(root / "GeoChat_Instruct/GeoChat_Instruct.json"),
                                      held_out, "geochat", geochat_images)
    vrs = read_json(root / "VRSBench/VRSBench_train.json")
    for filename in ["VRSBench_EVAL_Cap.json", "VRSBench_EVAL_vqa.json"]:
        for row in read_json(root / "VRSBench" / filename):
            vrs.append(dict(id=f"EVAL/{row['image_id']}/{row['question_id']}", image=row["image_id"],
                            conversations=[dict(from_="human", value="<image>\n" + row["question"]),
                                           dict(from_="gpt", value=row["ground_truth"])]))
            for turn in vrs[-1]["conversations"]:
                turn["from"] = turn.pop("from_")
    data["vrsbench_train"] = auxiliary(vrs, held_out, "vrsbench", vrsbench_images)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("data/prepared"))
    p.add_argument("--geochat-images", default="GeoChat_Instruct/images")
    p.add_argument("--vrsbench-images", default="VRSBench/images")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name, rows in build(a.root, a.geochat_images, a.vrsbench_images).items():
        with (a.output / (name + ".jsonl")).open("x") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")


if __name__ == "__main__":
    main()
