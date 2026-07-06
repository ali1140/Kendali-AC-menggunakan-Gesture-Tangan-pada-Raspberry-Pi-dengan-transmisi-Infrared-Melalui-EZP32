import argparse
import os
import cv2
import json
import glob
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RAW_DIR = os.path.join(BASE_DIR, "dataset_raw_img_20")
DST_DIR = os.path.join(BASE_DIR, "dataset_landmarks_20")

SEQ_LEN = 20
MIN_LEN = 10
MIN_PRESENCE = 0.8

WRIST_IDX      = 0
MIDDLE_MCP_IDX = 9




def calculate_features(lm_curr, lm_prev, handedness):

    raw_wrist_curr = lm_curr[WRIST_IDX].copy()
    raw_mid_curr   = lm_curr[MIDDLE_MCP_IDX].copy()
    palm_size = np.linalg.norm(raw_mid_curr - raw_wrist_curr) + 1e-6


    if lm_prev is None:
        vector_feats = np.zeros(4, dtype=np.float32)
    else:
        raw_wrist_prev = lm_prev[WRIST_IDX].copy()
        raw_mid_prev   = lm_prev[MIDDLE_MCP_IDX].copy()

        d_wrist = (raw_wrist_curr[:2] - raw_wrist_prev[:2]) / palm_size
        d_mid   = (raw_mid_curr[:2]   - raw_mid_prev[:2])   / palm_size

        if handedness == 'Left':
            d_wrist[0] *= -1
            d_mid[0]   *= -1

        vector_feats = np.concatenate([d_wrist, d_mid]) * 20.0


    rel_pos = lm_curr - raw_wrist_curr
    rel_pos = rel_pos / palm_size
    if handedness == 'Left': rel_pos[:, 0] *= -1
    flat_pos = rel_pos.reshape(-1)

    return np.concatenate([flat_pos, vector_feats])




def pad_sequence(features, target_len):

    current_len = len(features)

    if current_len >= target_len:

        return features[:target_len]


    diff = target_len - current_len


    last_feat = features[-1].copy()



    last_feat[-4:] = 0.0


    for _ in range(diff):
        features.append(last_feat)

    return features




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="classes", action="append", help="Nama kelas yang ingin diekstrak, contoh: --class Mode")
    parser.add_argument("--sample", dest="samples", action="append", help="Nama sample/folder raw tertentu yang ingin diekstrak")
    parser.add_argument("--only-new", action="store_true", help="Kompatibilitas command lama. Default script memang hanya ekstrak sample baru")
    parser.add_argument("--force", action="store_true", help="Ekstrak ulang sample walaupun file .npz sudah ada")
    parser.add_argument("--replace-class", action="store_true", help="Hapus output kelas target sebelum ekstraksi ulang")
    return parser.parse_args()


def process_dataset(target_classes=None, replace_class=False, target_samples=None, force=False):
    if not os.path.exists(RAW_DIR):
        print(f"[ERROR] {RAW_DIR} tidak ditemukan!")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=MIN_PRESENCE,
        min_tracking_confidence=MIN_PRESENCE
    )

    all_classes = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    if target_classes:
        missing = [class_name for class_name in target_classes if class_name not in all_classes]
        if missing:
            print(f"[ERROR] Kelas tidak ditemukan di {RAW_DIR}: {missing}")
            return
        classes = target_classes
    else:
        classes = all_classes

    print(f"=== EKSTRAKSI DENGAN SMART PADDING ===")
    print(f"Target Length: {SEQ_LEN}")
    print(f"Min Length   : {MIN_LEN} (Data pendek tetap diambil)")
    if target_classes:
        print(f"Kelas target : {classes}")
    else:
        print("Kelas target : Semua kelas")
    if force:
        print("Mode proses  : Force overwrite file yang sudah ada")
    else:
        print("Mode proses  : Skip file landmark yang sudah ada")
    print("-" * 50)

    total_samples = 0
    padded_samples = 0
    skipped_samples = 0
    skipped_existing = 0

    for class_name in classes:
        class_raw_path = os.path.join(RAW_DIR, class_name)
        class_dst_path = os.path.join(DST_DIR, class_name)
        if replace_class and target_classes and os.path.exists(class_dst_path):
            import shutil
            shutil.rmtree(class_dst_path)
        os.makedirs(class_dst_path, exist_ok=True)

        sample_folders = sorted([d for d in os.listdir(class_raw_path) if os.path.isdir(os.path.join(class_raw_path, d))])
        if target_samples:
            missing_samples = [sample_id for sample_id in target_samples if sample_id not in sample_folders]
            if missing_samples:
                print(f"[WARN] Sample tidak ditemukan di kelas '{class_name}': {missing_samples}")
            sample_folders = [sample_id for sample_id in sample_folders if sample_id in target_samples]

        print(f"Processing '{class_name}'...")

        for sample_id in sample_folders:
            sample_path = os.path.join(class_raw_path, sample_id)
            if not os.path.isdir(sample_path): continue

            save_path = os.path.join(class_dst_path, f"{sample_id}.npz")
            if os.path.exists(save_path) and not force:
                skipped_existing += 1
                continue

            img_files = sorted(glob.glob(os.path.join(sample_path, "*.jpg")))


            if len(img_files) < MIN_LEN:
                skipped_samples += 1
                continue

            features_seq = []
            lm_prev = None


            for fpath in img_files:
                frame = cv2.imread(fpath)
                if frame is None: break


                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    lm_obj = results.multi_hand_landmarks[0]
                    hand_lbl = results.multi_handedness[0].classification[0].label
                    lm_curr = np.array([[p.x, p.y, p.z] for p in lm_obj.landmark], dtype=np.float32)

                    feats = calculate_features(lm_curr, lm_prev, hand_lbl)
                    features_seq.append(feats)
                    lm_prev = lm_curr



            if len(features_seq) >= MIN_LEN:

                original_len = len(features_seq)
                features_seq = pad_sequence(features_seq, SEQ_LEN)


                data_arr = np.array(features_seq, dtype=np.float32)


                meta = {"class": class_name, "original_len": original_len, "padded": original_len < SEQ_LEN}
                np.savez_compressed(save_path, data=data_arr, meta=json.dumps(meta))

                total_samples += 1
                if original_len < SEQ_LEN:
                    padded_samples += 1
            else:
                skipped_samples += 1

    print("-" * 50)
    print("SELESAI.")
    print(f"Total Disimpan : {total_samples}")
    print(f"   - Asli (45+) : {total_samples - padded_samples}")
    print(f"   - Padding    : {padded_samples} (Data pendek yang diselamatkan)")
    print(f"Total Dibuang  : {skipped_samples} (Terlalu pendek/rusak)")
    print(f"Skip Existing : {skipped_existing}")

if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        target_classes=args.classes,
        replace_class=args.replace_class,
        target_samples=args.samples,
        force=args.force,
    )
