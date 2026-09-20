# Task 2 — 影像拼接品質指標、處理統計輸出與 Pandas DataFrame 回傳

## 目標

在既有的影像拼接（image stitching / mosaic）pipeline 完成後，新增一個統一的品質評估模組，用來計算並輸出下列 6 類指標：

1. **Reprojection Error**
2. **Inlier Ratio**
3. **Inlier Count**
4. **Cycle / Loop Error**
5. **Seam Error**
6. **Distortion**

除了品質指標之外，還必須記錄 **pipeline 執行統計資訊**，至少包含：

- 輸入影像總數
- 成功加入 mosaic 的影像數
- 失敗影像數
- 拼接成功率
- 失敗影像 index
- 總處理時間
- 平均每張影像處理時間
- pipeline 最終狀態

所有品質指標與處理統計必須整理成 **同一個 `pandas.DataFrame`**，由 pipeline 最後直接 `return`，方便後續：

- 模型 / matcher 比較
- 批次實驗
- CSV 儲存
- benchmark
- 視覺化
- 超參數調整

---

# 1. 核心需求

在完整 stitching pipeline 執行完畢後，呼叫：

```python
metrics_df = evaluate_stitching_metrics(...)
```

並回傳：

```python
return stitched_image, metrics_df
```

若目前 pipeline 原本只回傳：

```python
return stitched_image
```

請修改為：

```python
return stitched_image, metrics_df
```

不要只 `print()` 指標；必須保留數值並回傳 `pandas.DataFrame`。

---

# 2. DataFrame Schema

DataFrame 使用一個 summary row，同時包含兩類資訊：

1. **Stitching quality metrics**
2. **Pipeline process statistics**

至少產生以下 columns：

```python
[
    # Pipeline process statistics
    "pipeline_status",
    "input_image_count",
    "successful_image_count",
    "failed_image_count",
    "stitch_success_rate",
    "failed_image_indices",
    "total_processing_time_sec",
    "avg_processing_time_per_image_sec",

    # Stitching quality metrics
    "reprojection_error_px",
    "inlier_ratio",
    "inlier_count",
    "cycle_loop_error_px",
    "seam_error",
    "distortion",
]
```

DataFrame 最少包含一個 `summary` row：

```python
metrics_df = pd.DataFrame([
    {
        "pipeline_status": ...,
        "input_image_count": ...,
        "successful_image_count": ...,
        "failed_image_count": ...,
        "stitch_success_rate": ...,
        "failed_image_indices": ...,
        "total_processing_time_sec": ...,
        "avg_processing_time_per_image_sec": ...,
        "reprojection_error_px": ...,
        "inlier_ratio": ...,
        "inlier_count": ...,
        "cycle_loop_error_px": ...,
        "seam_error": ...,
        "distortion": ...,
    }
])
```

期望輸出概念如下：

```text
pipeline_status  input_image_count  successful_image_count  failed_image_count  stitch_success_rate  failed_image_indices  total_processing_time_sec  avg_processing_time_per_image_sec  reprojection_error_px  inlier_ratio  inlier_count  cycle_loop_error_px  seam_error  distortion
partial_success                 120                     113                   7               0.9417       [18, 44, 79, ...]                     38.52                             0.3210                   1.42        0.8074          3100                 6.21      0.0504      0.0831
```

---

# 3. Pipeline Process Statistics

除了影像品質指標外，必須記錄 pipeline 本身的執行結果。

## 3.1 `pipeline_status`

使用固定字串：

```text
success
partial_success
failed
```

定義：

- `success`：所有輸入影像都成功加入 final mosaic。
- `partial_success`：輸入影像大於 1 張時，至少 2 張成功加入同一個 mosaic，但不是所有影像都成功。
- `failed`：無法形成有效拼接。例如輸入影像大於 1 張，但最後只有 reference / anchor image，沒有任何其他影像成功加入 mosaic。
- 若輸入本來就只有 1 張，而且可正常輸出該影像，視為 `success`。

### Column

```python
"pipeline_status"
```

---

## 3.2 影像數量統計

### `input_image_count`

輸入 pipeline 的影像總數：

```python
input_image_count = len(images)
```

### `successful_image_count`

成功被註冊到 global mosaic coordinate system，且實際對 final mosaic 貢獻有效 pixel 的影像數量。

**計數規則必須固定：**

- reference / anchor image 如果有出現在 final mosaic 中，算成功 1 張。
- 只有 feature matching 成功但 transform validation 失敗，不算成功。
- transform 有算出來但最終沒有被加入 mosaic，不算成功。
- 同一張 image 不得重複計數。

### `failed_image_count`

```text
failed_image_count = input_image_count - successful_image_count
```

### `stitch_success_rate`

```text
stitch_success_rate = successful_image_count / input_image_count
```

如果：

```python
input_image_count == 0
```

則：

```python
stitch_success_rate = np.nan
```

範圍：

```text
0.0 ~ 1.0
```

### `failed_image_indices`

保存所有無法加入 final mosaic 的原始 image index，例如：

```python
[3, 7, 12]
```

DataFrame 中保留為 Python `list[int]` 即可。輸出成 `metrics.txt` 時，`DataFrame.to_string()` 會直接顯示該 list。

---

## 3.3 處理時間

使用：

```python
import time
start_time = time.perf_counter()
```

不要使用 `time.time()` 作為 benchmark timer。

### `total_processing_time_sec`

定義為：

> 從 stitching pipeline 開始執行，到 final mosaic 與所有 quality metrics 計算完成為止的 wall-clock elapsed time。process statistics 與 DataFrame 組裝本身只剩極小的 Python bookkeeping，不納入 benchmark。

包括：

- feature extraction / matching
- geometric verification
- transform estimation
- warping
- blending
- stitching metric calculation

預設**不包含**：

- caller 在進入 pipeline 前的檔案讀取時間
- final image / `metrics.txt` 寫入硬碟的時間

這樣不同 benchmark run 比較時定義一致。

計算：

```python
total_processing_time_sec = time.perf_counter() - start_time
```

### `avg_processing_time_per_image_sec`

```python
avg_processing_time_per_image_sec = (
    total_processing_time_sec / input_image_count
    if input_image_count > 0
    else np.nan
)
```

這個值只用來快速比較不同 pipeline 的整體 throughput，不等同於每張影像實際獨立耗時。

---

# 4. 指標定義

## 4.1 Reprojection Error

### 目的

衡量 feature correspondence 經過估計的 geometric transform 後，預測位置與實際 matched point 的距離。

對每一組影像 pair：

- source point：`p_i`
- destination point：`q_i`
- estimated transform：`H`

將 source point 投影：

```text
p'_i = project(H, p_i)
```

單點誤差：

```text
e_i = || p'_i - q_i ||_2
```

整體使用 **RMSE**：

```text
Reprojection Error = sqrt(mean(e_i^2))
```

### 要求

- 單位：`pixel`
- 只使用 **RANSAC inliers** 計算
- 不要把 outliers 算入 reprojection error
- 多組 image pair 時，將所有有效 inlier errors 合併後計算 global RMSE

### Column

```python
"reprojection_error_px"
```

### 趨勢

```text
越低越好
```

---

## 4.2 Inlier Ratio

### 目的

衡量 matcher 所找到的 matches 中，有多少比例符合目前估計的幾何模型。

定義：

```text
Inlier Ratio = Total Inliers / Total Matches
```

### 要求

如果有多組 image pair，不要直接對每個 pair 的 ratio 做普通平均。

應該使用：

```python
global_inlier_ratio = total_inlier_count / total_match_count
```

避免小樣本 pair 對結果造成過大的權重。

### Column

```python
"inlier_ratio"
```

### 數值範圍

```text
0.0 ~ 1.0
```

### 趨勢

```text
越高通常越好
```

---

## 4.3 Inlier Count

### 目的

表示通過 geometric verification 的有效 correspondence 數量。

多張影像時：

```text
Inlier Count = 所有成功配對 image pair 的 inlier 數總和
```

### Column

```python
"inlier_count"
```

### 型別

```python
int
```

### 趨勢

```text
通常越高越穩定，但必須搭配 Inlier Ratio 與 Reprojection Error 一起解讀。
```

---

## 4.4 Cycle / Loop Error

## 目的

衡量多張影像的 transformation chain 是否產生 global drift。

如果存在一個 loop：

```text
image A -> B -> C -> ... -> A
```

將所有 transformation composition：

```text
H_cycle = H_n_to_A @ ... @ H_B_to_C @ H_A_to_B
```

理想情況：

```text
H_cycle ≈ Identity Matrix
```

不要直接比較 Homography matrix element，因為 projective matrix 有 scale ambiguity。

### 計算方式

對 loop 起始影像建立 reference points，例如：

```python
[
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1],
    [width / 2, height / 2],
]
```

將這些 points 經過 `H_cycle` 投影後，計算回到原位置的 Euclidean pixel distance：

```text
cycle_error_i = || project(H_cycle, p_i) - p_i ||_2
```

每個 loop 使用 RMSE：

```text
Loop RMSE = sqrt(mean(cycle_error_i^2))
```

如果有多個 loop：

```text
Cycle / Loop Error = mean(all loop RMSE)
```

### Column

```python
"cycle_loop_error_px"
```

### 單位

```text
pixel
```

### 趨勢

```text
越低越好
```

### 沒有 loop 時

如果目前資料是單向 sequence，而且不存在 loop closure，不要偽造數字。

請回傳：

```python
np.nan
```

例如：

```python
cycle_loop_error_px = np.nan
```

---

## 4.5 Seam Error

## 目的

衡量 stitched image 在 seam 附近是否有明顯的亮度、顏色或局部結構不連續。

不要直接對整張 mosaic 計算 pixel difference。

只在實際 overlap / seam 附近評估。

## 建議計算方式

對於兩張已經 warp 到同一 mosaic coordinate system 的影像：

```python
warped_a
warped_b
mask_a
mask_b
```

先找 overlap：

```python
overlap = (mask_a > 0) & (mask_b > 0)
```

如果 pipeline 有 seam mask / seam path，優先使用真正 seam 周圍的 band，例如 seam 左右各 `5~10 px`。

如果目前沒有 seam path，可先使用 overlap region 作為 approximation。

將影像轉成 `float32`，範圍統一為 `[0, 1]`：

```python
img = img.astype(np.float32) / 255.0
```

計算 RGB mean absolute difference：

```text
Seam Error = mean(abs(warped_a - warped_b))
```

只統計有效 seam / overlap pixels。

### Column

```python
"seam_error"
```

### 建議範圍

若 normalized 至 `[0, 1]`：

```text
0.0 ~ 1.0
```

### 趨勢

```text
越低越好
```

### 注意

海面會受到：

- 波浪運動
- 太陽反光
- 水面非剛性變化
- 曝光變化

影響，因此 Seam Error 不等同於純幾何誤差，只能作為視覺連續性的輔助指標。

---

## 4.6 Distortion

## 目的

衡量 geometric warping 是否造成過度的局部形變。

只比較 homography matrix element 本身沒有明確物理意義，因此必須由 transformation 對局部幾何造成的影響來計算。

## 建議方法：Local Jacobian Anisotropy

對每一個 image-to-mosaic transform，在原圖建立固定 grid，例如：

```python
GRID_ROWS = 10
GRID_COLS = 10
```

在每一個 sampling point `(x, y)` 上，使用數值微分估計 warp function 的 Jacobian：

```text
J = [ dx'/dx   dx'/dy ]
    [ dy'/dx   dy'/dy ]
```

對 Jacobian 做 singular value decomposition：

```text
sigma_1 >= sigma_2 > 0
```

定義局部 anisotropic distortion：

```text
d = abs(log(sigma_1 / sigma_2))
```

理想情況：

```text
sigma_1 == sigma_2
=> d = 0
```

代表局部縮放在不同方向一致，沒有額外的 directional stretching。

最後：

```text
Distortion = mean(d)
```

對所有有效 grid points 與所有成功加入 mosaic 的影像計算平均。

### Column

```python
"distortion"
```

### 單位

```text
dimensionless
```

### 趨勢

```text
越接近 0 越好
```

### 無效位置

如果 Jacobian：

- singular
- 有 NaN / Inf
- `sigma_2 <= epsilon`
- 投影位置超出合理數值範圍

該 sampling point 必須忽略。

若所有 sampling points 均無法計算：

```python
np.nan
```

---

# 5. 建議程式架構

新增獨立模組：

```text
metrics.py
```

建議結構：

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reprojection_error(...):
    ...


def compute_inlier_statistics(...):
    ...


def compute_cycle_loop_error(...):
    ...


def compute_seam_error(...):
    ...


def compute_distortion(...):
    ...


def evaluate_stitching_metrics(...):
    ...
```

不要把所有 metric calculation 全部塞進主要 stitching function。

---

# 6. 建議的資料結構

在 stitching 過程中，要保留足夠的 intermediate data。

例如每一組 pair matching 結果：

```python
pair_result = {
    "src_index": 0,
    "dst_index": 1,
    "src_points": src_points,          # shape: [N, 2]
    "dst_points": dst_points,          # shape: [N, 2]
    "inlier_mask": inlier_mask,        # shape: [N]
    "homography": H,                   # 3 x 3
}
```

全部保存：

```python
pair_results: list[dict]
```

另外保留 image-to-global / image-to-mosaic transformations：

```python
global_transforms = {
    image_index: H_image_to_mosaic,
}
```

如有 loop closure：

```python
loops = [
    [0, 1, 2, 3, 0],
    [4, 5, 6, 4],
]
```

seam metric 至少需要：

```python
warped_images
warped_masks
```

如果 pipeline 本身有 seam finder，請額外保留：

```python
seam_masks
```

---

# 7. evaluate_stitching_metrics() Interface

實作一個統一入口，例如：

```python
def evaluate_stitching_metrics(
    pair_results,
    global_transforms,
    image_shapes,
    process_stats,
    warped_images=None,
    warped_masks=None,
    seam_masks=None,
    loops=None,
) -> pd.DataFrame:
    """
    Evaluate stitching quality, merge pipeline process statistics,
    and return a one-row pandas DataFrame.
    """
```

其中 `process_stats` 至少包含：

```python
process_stats = {
    "pipeline_status": pipeline_status,
    "input_image_count": input_image_count,
    "successful_image_count": successful_image_count,
    "failed_image_count": failed_image_count,
    "stitch_success_rate": stitch_success_rate,
    "failed_image_indices": failed_image_indices,
    "total_processing_time_sec": total_processing_time_sec,
    "avg_processing_time_per_image_sec": avg_processing_time_per_image_sec,
}
```

內部：

```python
metrics = {
    **process_stats,
    "reprojection_error_px": reprojection_error,
    "inlier_ratio": inlier_ratio,
    "inlier_count": int(inlier_count),
    "cycle_loop_error_px": cycle_loop_error,
    "seam_error": seam_error,
    "distortion": distortion,
}

metrics_df = pd.DataFrame([metrics])

return metrics_df
```

---

# 8. Pipeline 整合

假設原始 pipeline：

```python
def stitch_images(images):
    ...
    stitched_image = ...

    return stitched_image
```

修改成：

```python
def stitch_images(images):
    import time

    start_time = time.perf_counter()

    pair_results = []
    global_transforms = {}

    input_image_count = len(images)
    successful_image_indices = set()
    failed_image_indices = []

    ...

    # 當某張 image 經過 transform validation 並成功加入 mosaic 時：
    successful_image_indices.add(image_index)

    # 當確認某張 image 無法加入 mosaic 時：
    failed_image_indices.append(image_index)

    stitched_image = ...

    successful_image_count = len(successful_image_indices)
    failed_image_indices = sorted(set(failed_image_indices))
    failed_image_count = input_image_count - successful_image_count

    stitch_success_rate = (
        successful_image_count / input_image_count
        if input_image_count > 0
        else np.nan
    )

    if input_image_count == 0:
        pipeline_status = "failed"
    elif input_image_count == 1 and successful_image_count == 1:
        pipeline_status = "success"
    elif successful_image_count == input_image_count:
        pipeline_status = "success"
    elif input_image_count > 1 and successful_image_count >= 2:
        pipeline_status = "partial_success"
    else:
        pipeline_status = "failed"

    # quality metrics 可先算成 dict 或 intermediate values
    # 最後才建立 DataFrame，確保 total_processing_time_sec 包含 metric calculation。
    quality_metrics = compute_all_quality_metrics(...)

    total_processing_time_sec = time.perf_counter() - start_time
    avg_processing_time_per_image_sec = (
        total_processing_time_sec / input_image_count
        if input_image_count > 0
        else np.nan
    )

    process_stats = {
        "pipeline_status": pipeline_status,
        "input_image_count": input_image_count,
        "successful_image_count": successful_image_count,
        "failed_image_count": failed_image_count,
        "stitch_success_rate": stitch_success_rate,
        "failed_image_indices": failed_image_indices,
        "total_processing_time_sec": total_processing_time_sec,
        "avg_processing_time_per_image_sec": avg_processing_time_per_image_sec,
    }

    metrics_df = build_metrics_dataframe(
        process_stats=process_stats,
        quality_metrics=quality_metrics,
    )

    return stitched_image, metrics_df
```

呼叫端：

```python
stitched_image, metrics_df = stitch_images(images)

print(metrics_df)

# metrics.txt 必須與結果圖片位於同一個 directory
save_metrics_txt(metrics_df, result_image_path)
```

---

# 9. Metrics 檔案輸出

除了回傳 `pandas.DataFrame` 之外，pipeline **必須**在結果圖片所在的相同目錄自動建立：

```text
metrics.txt
```

例如結果圖片為：

```text
output_dir/stitched.png
```

則輸出結構必須至少為：

```text
output_dir/
├── stitched.png
└── metrics.txt
```

`metrics.txt` 的內容必須直接呈現 pipeline 最後 return 的同一份 `metrics_df`，不得另外重新計算一份 metrics，以避免檔案內容與 return value 不一致。

建議實作：

```python
from pathlib import Path


def save_metrics_txt(metrics_df: pd.DataFrame, result_image_path: str | Path) -> Path:
    result_image_path = Path(result_image_path)
    metrics_path = result_image_path.parent / "metrics.txt"

    metrics_path.write_text(
        metrics_df.to_string(index=False),
        encoding="utf-8",
    )

    return metrics_path
```

輸出的 `metrics.txt` 例如：

```text
pipeline_status  input_image_count  successful_image_count  failed_image_count  stitch_success_rate  failed_image_indices  total_processing_time_sec  avg_processing_time_per_image_sec  reprojection_error_px  inlier_ratio  inlier_count  cycle_loop_error_px  seam_error  distortion
partial_success                 120                     113                   7               0.9417       [18, 44, 79, ...]                     38.52                             0.3210                   1.42        0.8074          3100                 6.21      0.0504      0.0831
```

如果 metric 無法計算，必須與 DataFrame 一樣保留 `NaN`：

```text
pipeline_status  input_image_count  successful_image_count  failed_image_count  stitch_success_rate  failed_image_indices  total_processing_time_sec  avg_processing_time_per_image_sec  reprojection_error_px  inlier_ratio  inlier_count  cycle_loop_error_px  seam_error  distortion
partial_success                 120                     113                   7               0.9417       [18, 44, 79, ...]                     38.52                             0.3210                   1.42        0.8074          3100                  NaN      0.0504      0.0831
```

### Pipeline 整合範例

```python
stitched_image, metrics_df = stitch_images(images)

cv2.imwrite(str(result_image_path), stitched_image)
save_metrics_txt(metrics_df, result_image_path)

return stitched_image, metrics_df
```

### 必要要求

- `metrics.txt` 必須與最終結果圖片位於**同一個 directory**。
- 檔名固定使用 `metrics.txt`。
- 內容來源必須是最後 return 的同一個 `metrics_df`。
- 使用 `metrics_df.to_string(index=False)`，方便直接用文字編輯器閱讀。
- 每次 pipeline 執行可覆寫該次輸出目錄中的舊 `metrics.txt`。
- 寫入 `metrics.txt` 失敗時應回報清楚的 I/O error；不得靜默忽略。
- 建立 `metrics.txt` **不能取代** DataFrame return。

最終仍然必須：

```python
return stitched_image, metrics_df
```

## 9.1 CSV 輸出（可選）

若需要機器讀取或後續批次分析，DataFrame 仍應可直接：

```python
metrics_df.to_csv("stitching_metrics.csv", index=False)
```

CSV 可作為額外輸出，例如：

```text
output_dir/
├── stitched.png
├── metrics.txt
└── stitching_metrics.csv
```

但目前此 Task 的必要檔案輸出是 `metrics.txt`；CSV 為額外選項。

---

# 10. NaN / Failure Handling

任何單一 metric 無法計算時，不允許讓整個 stitching pipeline crash。

使用：

```python
np.nan
```

例如：

```python
{
    "reprojection_error_px": 1.52,
    "inlier_ratio": 0.79,
    "inlier_count": 2841,
    "cycle_loop_error_px": np.nan,
    "seam_error": 0.061,
    "distortion": 0.084,
}
```

特別是：

- 沒有 loop closure → `cycle_loop_error_px = np.nan`
- 沒有有效 overlap → `seam_error = np.nan`
- 沒有有效 transform → `distortion = np.nan`
- 沒有 valid inlier → `reprojection_error_px = np.nan`
- total matches = 0 → `inlier_ratio = np.nan`
- input image count = 0 → `stitch_success_rate = np.nan`
- input image count = 0 → `avg_processing_time_per_image_sec = np.nan`

`input_image_count`、`successful_image_count`、`failed_image_count` 屬於 process statistics，不應使用 `NaN` 取代可明確計數的結果。

不要用 `0` 代表「無法計算」，因為對品質 metric 而言，`0` 本身可能代表完美結果。

---

# 11. Numerical Stability

所有 geometric calculations：

```python
np.float64
```

避免長 transformation chain 使用 `float32` 累積誤差。

Homography normalization：

```python
if abs(H[2, 2]) > 1e-12:
    H = H / H[2, 2]
```

projection denominator：

```python
abs(w) < 1e-12
```

視為 invalid point。

計算完成前檢查：

```python
np.isfinite(...)
```

---

# 12. 額外建議：保留 Pair-Level Metrics

除了最終 summary DataFrame，建議同時建立 pair-level DataFrame，方便 debugging。

例如：

```text
src_index | dst_index | match_count | inlier_count | inlier_ratio | reprojection_error_px
0         | 1         | 2350        | 1910         | 0.8128       | 1.23
1         | 2         | 1982        | 1430         | 0.7215       | 1.87
2         | 3         | 2844        | 2301         | 0.8091       | 1.41
```

可以實作：

```python
summary_df, pair_metrics_df = evaluate_stitching_metrics(...)
```

但如果要維持目前需求的簡單 interface，至少一定要 return `summary_df`。

建議主要 pipeline 最終仍使用：

```python
return stitched_image, metrics_df
```

pair-level DataFrame 可以另外儲存：

```text
pair_metrics.csv
```

---

# 13. Logging

pipeline 完成時輸出：

```text
=== Stitching Summary ===
Pipeline Status    : partial_success
Input Images       : 120
Successful Images  : 113
Failed Images      : 7
Success Rate       : 94.17%
Failed Indices     : [18, 44, 79, ...]
Total Time         : 38.52 sec
Avg Time / Image   : 0.3210 sec

=== Stitching Metrics ===
Reprojection Error : 1.42 px
Inlier Ratio       : 0.8074
Inlier Count       : 3100
Cycle / Loop Error : 6.21 px
Seam Error         : 0.0504
Distortion         : 0.0831
=========================
```

如果 metric 是 NaN：

```text
Cycle / Loop Error : N/A
```

但 logging 只是方便閱讀，實際資料仍以 DataFrame 為準。

---

# 14. Unit Tests

至少新增以下測試。

## Test 1 — Identity Homography

```python
H = np.eye(3)
```

source points 與 destination points 完全相同。

期望：

```text
reprojection_error_px ≈ 0
```

---

## Test 2 — Known Translation

建立：

```text
x' = x + 10
y' = y + 20
```

如果 destination points 正確套用 translation：

```text
reprojection_error_px ≈ 0
```

---

## Test 3 — Inlier Ratio

```python
match_count = 100
inlier_count = 80
```

期望：

```text
inlier_ratio = 0.8
```

---

## Test 4 — Identity Cycle

```text
A -> B -> C -> A
```

transform composition 為 identity。

期望：

```text
cycle_loop_error_px ≈ 0
```

---

## Test 5 — No Loop

```python
loops = []
```

期望：

```python
np.isnan(cycle_loop_error_px)
```

---

## Test 6 — Identical Overlap

兩張 overlap image 完全相同。

期望：

```text
seam_error ≈ 0
```

---

## Test 7 — Identity Warp Distortion

```python
H = np.eye(3)
```

期望：

```text
distortion ≈ 0
```

---

## Test 8 — metrics.txt Output

建立暫存輸出目錄與測試用 `metrics_df`，並假設結果圖片路徑為：

```text
/tmp/stitch_test/stitched.png
```

呼叫：

```python
metrics_path = save_metrics_txt(metrics_df, result_image_path)
```

期望：

```text
/tmp/stitch_test/metrics.txt
```

必須存在，且：

```python
metrics_path.read_text(encoding="utf-8") == metrics_df.to_string(index=False)
```

---

## Test 9 — Process Statistics

假設：

```python
input_image_count = 10
successful_image_indices = {0, 1, 2, 3, 4, 5, 6, 8}
failed_image_indices = [7, 9]
```

期望：

```text
successful_image_count = 8
failed_image_count = 2
stitch_success_rate = 0.8
pipeline_status = "partial_success"
```

另外測試空輸入：

```python
input_image_count = 0
```

期望：

```python
pipeline_status == "failed"
np.isnan(stitch_success_rate)
np.isnan(avg_processing_time_per_image_sec)
```

再測試「只有 anchor，沒有真正拼接成功」：

```python
input_image_count = 10
successful_image_count = 1
```

期望：

```python
pipeline_status == "failed"
```

---

# 15. Acceptance Criteria

此 Task 完成必須滿足：

- [ ] stitching pipeline 可正常完成影像拼接
- [ ] DataFrame 同時包含 quality metrics 與 process statistics
- [ ] 記錄 `pipeline_status`
- [ ] 記錄 `input_image_count`
- [ ] 記錄 `successful_image_count`
- [ ] 記錄 `failed_image_count`
- [ ] 記錄 `stitch_success_rate`
- [ ] 記錄 `failed_image_indices`
- [ ] 記錄 `total_processing_time_sec`
- [ ] 記錄 `avg_processing_time_per_image_sec`
- [ ] 成功影像的計數規則符合本文件定義
- [ ] 使用 `time.perf_counter()` 計時
- [ ] pipeline 完成後自動計算 Reprojection Error
- [ ] 自動計算 Inlier Ratio
- [ ] 自動計算 Inlier Count
- [ ] 自動計算 Cycle / Loop Error
- [ ] 自動計算 Seam Error
- [ ] 自動計算 Distortion
- [ ] 所有 metric 使用明確且一致的定義
- [ ] 所有 metric 被整理進 `pandas.DataFrame`
- [ ] pipeline 使用 `return stitched_image, metrics_df`
- [ ] 無法計算的 metric 使用 `np.nan`
- [ ] 不得因單一 metric 無法計算導致 stitching pipeline crash
- [ ] 最終結果圖片所在 directory 會自動產生 `metrics.txt`
- [ ] `metrics.txt` 的內容與 return 的 `metrics_df` 完全相同
- [ ] `metrics.txt` 使用 `metrics_df.to_string(index=False)` 輸出
- [ ] DataFrame 可直接輸出成 CSV（可選）
- [ ] 至少完成上述基本 unit tests
- [ ] 所有幾何計算注意 numerical stability

---

# 16. 最終預期使用方式

```python
stitched_image, metrics_df = stitch_images(images)

cv2.imwrite(str(result_image_path), stitched_image)
save_metrics_txt(metrics_df, result_image_path)

print(metrics_df)

metrics_df.to_csv(
    "outputs/stitching_metrics.csv",
    index=False,
)
```

輸出：

```text
  pipeline_status  input_image_count  successful_image_count  failed_image_count  stitch_success_rate failed_image_indices  total_processing_time_sec  avg_processing_time_per_image_sec  reprojection_error_px  inlier_ratio  inlier_count  cycle_loop_error_px  seam_error  distortion
0 partial_success                120                     113                   7               0.9417       [18, 44, 79]                     38.52                             0.3210                   1.42        0.8074          3100                 6.21      0.0504      0.0831
```

此 DataFrame 將同時描述「拼得好不好」與「pipeline 執行得穩不穩、快不快」，作為後續比較不同 stitching pipeline、feature matcher、geometric model 與參數設定的統一 benchmark 輸出。
