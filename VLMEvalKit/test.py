import pandas as pd



def align_mmbench(file_path):
    l3_to_l2_l1 = {
        "action_recognition": ("attribute_reasoning", "Reasoning"),
        "attribute_comparison": ("attribute_reasoning", "Reasoning"),
        "attribute_recognition": ("attribute_reasoning", "Reasoning"),
        "celebrity_recognition": ("finegrained_perception (instance-level)", "Perception"),
        "function_reasoning": ("attribute_reasoning", "Reasoning"),
        "future_prediction": ("logic_reasoning", "Reasoning"),
        "identity_reasoning": ("relation_reasoning", "Reasoning"),
        "image_emotion": ("coarse_perception", "Perception"),
        "image_quality": ("coarse_perception", "Perception"),
        "image_scene": ("coarse_perception", "Perception"),
        "image_style": ("coarse_perception", "Perception"),
        "image_topic": ("coarse_perception", "Perception"),
        "nature_relation": ("relation_reasoning", "Reasoning"),
        "object_localization": ("finegrained_perception (instance-level)", "Perception"),
        "ocr": ("coarse_perception", "Perception"),
        "physical_property_reasoning": ("attribute_reasoning", "Reasoning"),
        "physical_relation": ("relation_reasoning", "Reasoning"),
        "social_relation": ("relation_reasoning", "Reasoning"),
        "spatial_relationship": ("attribute_reasoning", "Reasoning"),
        "structuralized_imagetext_understanding": ("logic_reasoning", "Reasoning"),
    }

    # 엑셀 로드
    df = pd.read_excel(file_path)

    # 정답 비교
    df["correct"] = df["answer"] == df["prediction"]

    # L2, L1 컬럼 생성
    df["L2_category"] = df["category"].map(lambda x: l3_to_l2_l1[x][0])
    df["L1_category"] = df["category"].map(lambda x: l3_to_l2_l1[x][1])

    # 통계 계산 함수
    def compute_stats(field):
        return df.groupby(field)["correct"].agg(
            correct_count="sum", total_count="count", accuracy="mean"
        ).reset_index()

    # L1, L2, L3 테이블 생성
    l1_table = compute_stats("L1_category")
    l2_table = compute_stats("L2_category")
    l3_table = compute_stats("category")

    total = len(df)
    correct = df["correct"].sum()
    overall_accuracy = correct / total * 100 if total > 0 else 0
    overall_df = pd.DataFrame({
        "total": [total],
        "correct": [correct],
        "accuracy (%)": [overall_accuracy]
    })

    # 저장
    output_path = "MMBench_L1_L2_L3_Accuracy.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        l1_table.to_excel(writer, index=False, sheet_name="L1")
        l2_table.to_excel(writer, index=False, sheet_name="L2")
        l3_table.to_excel(writer, index=False, sheet_name="L3")
        overall_df.to_excel(writer, index=False, sheet_name="Overall")

    print(f"✅ Saved : {output_path}")



def align_mme(file_path):
    df = pd.read_excel(file_path)

    def is_correct(row):
        return str(row['answer']).strip().lower() == str(row['prediction']).strip().lower()

    df['is_correct'] = df.apply(is_correct, axis=1)

    total = len(df)
    correct = df['is_correct'].sum()
    overall_accuracy = correct / total * 100 if total > 0 else 0


    category_stats = df.groupby('category')['is_correct'].agg(['sum', 'count']).reset_index()
    category_stats.columns = ['category', 'matched (correct)', 'total']
    category_stats['accuracy (%)'] = category_stats['matched (correct)'] / category_stats['total'] * 100

    output_path = "MME_Accuracy.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        category_stats.to_excel(writer, index=False, sheet_name='CategoryStats')
        pd.DataFrame({
            'total': [total],
            'correct': [correct],
            'accuracy (%)': [overall_accuracy]
        }).to_excel(writer, index=False, sheet_name='Overall')

    print(f"\n✅ Saved : {output_path}")
    return category_stats

def merge_csvs(csv_paths, output_excel_path="merged_results.xlsx", sheet_name_prefix="sheet"):
    merged_df = pd.DataFrame()

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            df['__source__'] = path
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        except Exception as e:
            raise ValueError

    merged_df.to_excel(output_excel_path, index=False)
    print(f"saved : {output_excel_path}")


def merge_jsons(json_paths, output_excel_path="merged_results.xlsx", sheet_name="merged"):
    merged_rows = []

    for path in json_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            row = {}

            bleu = data.get("Bleu", [None] * 4)
            for i, score in enumerate(bleu, start=1):
                row[f"Bleu_{i}"] = score

            row["ROUGE_L"] = data.get("ROUGE_L", None)
            row["CIDEr"] = data.get("CIDEr", None)

            row["source"] = os.path.dirname(path)

            merged_rows.append(row)

        except Exception as e:
            print(f"Error : {path} : {e}")

    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_excel(output_excel_path, sheet_name=sheet_name, index=False)
    print(f"Saved : {output_excel_path}")



csv_true_json_false = False

if __name__ == "__main__":
    
    import os, json

    if csv_true_json_false:
        low_path = "output/mmbench/unskip_012_293031_i"
        high_path = "llava-1.5-7b/llava-1.5-7b_MME_score.csv"
        paths = []
        names = []

        for i in range(3, 29, 1):
            full_path = os.path.join(low_path, f"{i}", high_path)
            name = f"{low_path}/{i}"
            print(full_path)
            print(name)
            paths.append(full_path)

        merge_csvs(paths)
    else:
        low_path = "output/coco/100_i"
        high_path = "llava-1.5-7b/llava-1.5-7b_COCO_VAL_score.json"
        paths = []
        names = []
        for i in range(3,29,1):
            full_path = os.path.join(low_path, f"{i}", high_path)
            name = f"{low_path}/{i}"
            print(full_path)
            print(name)
            paths.append(full_path)
            print(paths)

        merge_jsons(paths)
