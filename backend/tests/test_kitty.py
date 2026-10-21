from tools.kitti import parse_label_file, KittiDataset, meets_difficulty, KittiDataset, load_split, fallback_split, get_splits, meets_difficulty, parse_label_file
from tools.types import Box, GTObject
import pytest

from PIL import Image


SAMPLE = """\
Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
Pedestrian 0.50 2 -0.20 10 20 30 60 1.8 0.5 0.9 1 1 10 0
DontCare -1 -1 -10 100 100 200 150 -1 -1 -1 -1000 -1000 -1000 -10
this line is garbage
"""

LEVELS = ("easy", "moderate", "hard")


def test_parse_label_file(tmp_path):
    f = tmp_path / "000000.txt"
    f.write_text(SAMPLE)
    objs = parse_label_file(f)

    assert len(objs) == 3                      # garbage line skipped
    assert objs[0].cls == "Car"
    assert objs[0].box == Box(587.01, 173.33, 614.12, 200.12)
    assert objs[1].truncated == 0.5 and objs[1].occluded == 2
    assert objs[2].cls == "DontCare" and objs[2].truncated == -1


def _gt(height, trunc=0.0, occ=0):
    return GTObject("Car", Box(0, 0, 50, height), trunc, occ)


def _levels(g):
    return [meets_difficulty(g, d) for d in LEVELS]


def test_clear_close_car_meets_all_levels():
    assert _levels(_gt(50, 0.10, 0)) == [True, True, True]


def test_small_partly_occluded_is_not_easy():
    assert _levels(_gt(30, 0.20, 1)) == [False, True, True]


def test_small_cut_off_hidden_is_hard_only():
    assert _levels(_gt(26, 0.40, 2)) == [False, False, True]


def test_tiny_object_meets_no_level():
    assert _levels(_gt(20)) == [False, False, False]


def test_dataset_accepts_both_root_forms(tmp_path):
    train = tmp_path / "training"
    (train / "image_2").mkdir(parents=True)
    (train / "label_2").mkdir()
    Image.new("RGB", (100, 50)).save(train / "image_2" / "000000.png")
    (train / "label_2" / "000000.txt").write_text("Car 0 0 0 1 2 3 4 0 0 0 0 0 0 0\n")

    for root in (tmp_path, train):
        ds = KittiDataset(root)
        assert ds.ids == ["000000"] and len(ds) == 1
        assert ds.load_image("000000").size == (100, 50)
        assert ds.gts("000000")[0].box == Box(1, 2, 3, 4)

def _make_dataset(tmp_path, ids=("000000", "000001", "000002")):
    label_dir = tmp_path / "training" / "label_2"
    label_dir.mkdir(parents=True)
    for i in ids:
        (label_dir / f"{i}.txt").write_text("Car 0 0 0 1 2 3 4 0 0 0 0 0 0 0\n")
    return KittiDataset(tmp_path)


def _write_split(tmp_path, name, text):
    folder = tmp_path / "ImageSets"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.txt").write_text(text)


def test_load_split_reads_ids(tmp_path):
    _write_split(tmp_path, "train", "000000\n000002\n\n")
    assert load_split(tmp_path, "train") == ["000000", "000002"]


def test_load_split_missing_returns_none(tmp_path):
    assert load_split(tmp_path, "train") is None


def test_fallback_split_is_complete_disjoint_and_repeatable():
    ids = [f"{i:06d}" for i in range(20)]
    with pytest.warns(UserWarning):
        train, val = fallback_split(ids)
    assert sorted(train + val) == ids and not set(train) & set(val)
    with pytest.warns(UserWarning):
        assert fallback_split(ids) == (train, val)


def test_get_splits_uses_imagesets_and_drops_unknown_ids(tmp_path):
    ds = _make_dataset(tmp_path)
    _write_split(tmp_path, "train", "000000\n000002\n")
    _write_split(tmp_path, "val", "000001\n999999\n")     # 999999 is not in the dataset
    assert get_splits(ds) == (["000000", "000002"], ["000001"])


def test_get_splits_falls_back_with_warning(tmp_path):
    ds = _make_dataset(tmp_path)
    with pytest.warns(UserWarning):
        train, val = get_splits(ds)
    assert sorted(train + val) == ds.ids