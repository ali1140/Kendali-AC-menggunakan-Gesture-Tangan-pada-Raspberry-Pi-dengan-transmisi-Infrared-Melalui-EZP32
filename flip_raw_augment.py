"""
flip_raw_augment.py
--------------------
Menggandakan dataset raw image dengan membalik (flip horizontal) semua frame.
Hasil flip disimpan di folder kelas yang sama sebagai sampel baru dengan suffix '_flip'.
Sampel asli TIDAK dihapus atau diubah.

Cara pakai:
    python flip_raw_augment.py
"""

import os
import cv2
import glob
import shutil




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset_raw_img"))


TARGET_CLASSES = ["Mode"]





def count_samples(class_path):
    """Hitung jumlah sub-folder (sampel) di dalam folder kelas."""
    if not os.path.exists(class_path):
        return 0
    return len([d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))])


def flip_class(class_name):
    """Flip horizontal semua frame di setiap sampel untuk satu kelas."""
    class_path = os.path.join(RAW_DIR, class_name)

    if not os.path.exists(class_path):
        print(f"  [SKIP] Folder '{class_name}' tidak ditemukan di {RAW_DIR}")
        return 0

    sample_folders = sorted([
        d for d in os.listdir(class_path)
        if os.path.isdir(os.path.join(class_path, d)) and not d.endswith("_flip")
    ])

    created = 0
    skipped = 0

    for sample_id in sample_folders:
        src_path = os.path.join(class_path, sample_id)
        dst_id = f"{sample_id}_flip"
        dst_path = os.path.join(class_path, dst_id)


        if os.path.exists(dst_path):
            skipped += 1
            continue

        os.makedirs(dst_path, exist_ok=True)

        img_files = sorted(glob.glob(os.path.join(src_path, "*.jpg")))

        if len(img_files) == 0:
            shutil.rmtree(dst_path, ignore_errors=True)
            continue

        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is None:
                continue

            flipped = cv2.flip(img, 1)

            filename = os.path.basename(img_path)
            cv2.imwrite(os.path.join(dst_path, filename), flipped)

        created += 1

    return created, skipped


def main():
    if not os.path.exists(RAW_DIR):
        print(f"[ERROR] Folder dataset raw tidak ditemukan: {RAW_DIR}")
        return

    print("=" * 50)
    print("  FLIP AUGMENTATION - RAW IMAGES")
    print("=" * 50)
    print(f"Sumber  : {RAW_DIR}")
    print(f"Target  : {', '.join(TARGET_CLASSES)}")
    print("-" * 50)


    print("\n📊 SEBELUM FLIP:")
    for cls in TARGET_CLASSES:
        cls_path = os.path.join(RAW_DIR, cls)
        cnt = count_samples(cls_path)
        print(f"  {cls:15s} : {cnt} sampel")


    print("\n🔄 Memproses flip...")
    for cls in TARGET_CLASSES:
        print(f"\n  [{cls}]")
        created, skipped = flip_class(cls)
        print(f"    ✅ Dibuat baru : {created} sampel flip")
        if skipped > 0:
            print(f"    ⏭️  Dilewati   : {skipped} (sudah ada versi _flip)")


    print("\n" + "-" * 50)
    print("📊 SESUDAH FLIP:")
    for cls in TARGET_CLASSES:
        cls_path = os.path.join(RAW_DIR, cls)
        cnt = count_samples(cls_path)
        print(f"  {cls:15s} : {cnt} sampel")


    print("\n" + "=" * 50)
    print("  DATASET BALANCE (SEMUA KELAS)")
    print("=" * 50)
    all_classes = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ])
    max_count = 0
    counts = {}
    for cls in all_classes:
        cnt = count_samples(os.path.join(RAW_DIR, cls))
        counts[cls] = cnt
        if cnt > max_count:
            max_count = cnt

    for cls, cnt in counts.items():
        if cnt == 0:
            continue
        bar = "█" * int(cnt / max(max_count, 1) * 25)
        status = " ⚠️ LOW" if max_count > 0 and cnt < max_count * 0.6 else ""
        print(f"  {cls:15s} : {cnt:4d}  {bar}{status}")

    total = sum(counts.values())
    print(f"  {'TOTAL':15s} : {total:4d}")
    print("=" * 50)
    print("\n✅ Selesai! Sekarang jalankan extract → augmentasi → train.")


if __name__ == "__main__":
    main()
