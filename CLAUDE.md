# Drone Sea-Surface Mosaic Pipeline

無人機垂直下視海面影像的拼接 pipeline。最終產出 georeferenced mosaic
與一份 pandas DataFrame 形式的品質指標。

## 環境
- Python 3.12.3（系統路徑 /usr/bin/python3，沒有 python 這個 alias，只能用 python3）
- Poetry 2.5.1，位於 ~/.local/bin/poetry
- GPU：4 張 NVIDIA TITAN RTX 24GB，Driver 580.126.18，CUDA 13.0
  - **GPU 1 目前被其他使用者佔用約 20GB，不要用**
  - 一律用 GPU 0 / 2 / 3，跑任何 GPU 相關指令前先 nvidia-smi 確認狀態
  - 用法：CUDA_VISIBLE_DEVICES=0 python3 your_script.py
- 這是共用的容器化平台（Kubernetes pod），沒有 sudo、沒有 apt 安裝權限、
  沒有 Docker（也沒有 podman/apptainer 替代品）
- 依賴一律用 poetry add，不要用 pip install，不要用 --break-system-packages
- Dockerfile / compose.yaml 是這個專案的交付物，用來證明可重現性，
  但無法在這台開發機上實際 build 或驗證，寫的時候要更嚴謹地照規格來

## 硬性架構約束
1. 不准使用 cv2.Stitcher_create() 或任何黑盒 stitcher，
   pipeline 拆成 estimate → compose → warp → blend 四段
2. 必須保留 pair_results、global_transforms、warped_images、warped_masks
3. Matcher 走 protocol，不要把某個 matcher 寫死在 pipeline 裡
4. 全域 transform 不要用 pairwise homography 連乘，
   要當 pose graph 並以 GPS 位置為 anchor 做最佳化
5. 算不出來的 metric 填 np.nan，不要用 0

## 驗收標準
docs/task2.md 是這個專案的 metrics 規格書，也是驗收標準。
動到 metrics.py 或 pipeline.py 之前先讀它。

## 執行限制
- 不要主動跑完整資料集（上千張影像，會跑很久也會吃滿 GPU）
- 需要 GPU 或超過一分鐘的指令，先問我
- 測試一律用 tests/fixtures/ 裡的 synthetic 資料或資料集的小子集

## 已知的暫緩事項
- geo/camera.py、geo/direct.py：目前 data/smoke/ 是網路上找到的替代
  樣本（DJI ZH20T，Estepona/Gibraltar 一帶），相機規格細節不可靠
  （感光元件尺寸靠 1/2.3" 規格推算、無畸變係數、高度基準有 AbsoluteAltitude
  vs RelativeAltitude 約 50m 落差未釐清）。且 10 月會拿到正式資料集，
  屆時 metadata 格式可能完全不同。決定暫緩精確的 direct georeferencing，
  改用 GPS 座標僅作為 pose graph 的軟性錨點（見 posegraph.py 的
  GPSAnchor），不追求絕對地理精度與真實比例尺（GSD）。
  10 月拿到正式資料與規格後，重新評估是否完整實作這兩個模組。
- 這個決定的影響：拼接結果會是像素空間的 mosaic，相對位置關係正確，
  但無法回答「這張圖對應實際多少平方公尺」或跟外部資料集做絕對座標比對。
- estimate.py 的 `estimate_all_pairs(pairs=...)`：`pairs` 參數的長期設計意圖是接上
  GPS 距離篩選（用 geo/projection.py 的局部平面座標算候選對），取代窮舉全配對，
  但這套通用邏輯留到 10 月接上正式資料集、metadata 格式底定後再做。目前
  smoke test（data/smoke/ 10 張連續飛行序列）用 `sequential_pairs()` 明確傳入
  相鄰配對 `(i, i+1)`，只是為了先驗證 pipeline 跑得通；`pairs=None` 仍維持
  「窮舉全配對」的字面語意，不會被這個 smoke test 策略偷偷取代。

## 已知的限制
- 海面影像的 SIFT inlier 密度天生偏低，這是資料本身的限制，不是實作問題。
  用 `data/smoke/` 兩張相鄰影像（0352 vs 0353）做過一次對照實驗：
  - `cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)`：match_count=11283，
    inlier_count=7，inlier_ratio≈0.0006（幾乎全是雜訊，crossCheck 對這批
    影像不適用，會接受大量模糊匹配）。
  - `cv2.BFMatcher(cv2.NORM_L2).knnMatch(k=2)` + Lowe's ratio test
    （ratio=0.75）：match_count=289，inlier_count=40，inlier_ratio≈0.1384
    （比 crossCheck 好 230 倍，但仍遠低於一般陸地場景常見的 >0.3）。
  - 結論：正式的 SIFT matcher 實作定案用 knnMatch + ratio test，不要用
    crossCheck。但即使排除 crossCheck 的雜訊，inlier_ratio 仍只有 13.8%，
    代表海面紋理重複性高（波浪造成大量相似 descriptor）本身就會限制
    feature-based matching 的可靠對應點密度，這是真實存在的限制，不是
    matcher 調參可以完全解決的問題。
  - 這正是本專案採用「feature matching + GPS anchor 兜底」混合式架構的
    理由（見上面「已知的暫緩事項」與 posegraph.py 的 GPSAnchor）：純靠
    feature matching 建出的 pose graph 在低 inlier 密度時連通性/穩健度
    不足，需要 GPS 軟性錨點補強。
  - 這組數字（13.8% baseline inlier_ratio，0352 vs 0353）可以當作之後
    評估是否要導入 RoMa 等 dense matcher 的比較基準線。
  - 正式的 SIFT matcher（含 ratio test）與是否要導入 RoMa，留到
    feature-based pipeline 任務時再一起做，目前不動手實作。
- **compose_global_transforms 的 GPS anchor 機制，實質上依賴某種粗略的
  像素↔公尺換算才能發揮拉力，這個依賴之前沒有被意識到。已選定方向 A：
  用針孔相機近似算一個粗略的 pixels_per_meter，跟完整的
  `geo/camera.py`/`geo/direct.py` 精確 georeferencing 做清楚區隔——這是
  兩件不同精度需求的事：compose.py 只需要「大概對得上量級」，direct
  georeferencing 需要「準確的地理座標」。** 用真實資料實測過（不是合成
  猜測值），過程與結論如下：
  - **pixels_per_meter 換算方式（已定案）**：用 DFOV（對角線視角）+ 飛行
    高度算地面對角線覆蓋長度，再依影像寬高比把對角線覆蓋拆成
    ground_width_m / ground_height_m，`pixels_per_meter = 對角線像素數 /
    對角線地面覆蓋公尺數`。用這批影像的真實數字：
    `RelativeAltitude≈99.978m`、`DFOV=82.9°`（DJI H20T 官方規格）、
    `4056×3040`，算出 `ground_diagonal≈176.6m`（`ground_width≈141.3m`、
    `ground_height≈105.9m`），`pixels_per_meter≈28.703`
    （`meters_per_pixel≈0.03484`）。
  - **這是粗糙估計，近似來源要跟精確 georeferencing 明確區隔**：
    (1) 針孔相機模型，沒有實際相機內參標定；(2) 用 `RelativeAltitude`
    當高度基準，`AbsoluteAltitude` vs `RelativeAltitude` 的落差爭議見上面
    「已知的暫緩事項」，還沒有釐清哪個基準更可靠；(3) 沒有做鏡頭畸變校正。
  - **驗證結果**：套用 `pixels_per_meter≈28.703` 後，0352→0353 這對真實
    邊的「原始（未加權）」edge/anchor 殘差比從 ≈170x（2.2 個數量級）降到
    ≈5.9x（0.77 個數量級）——單位不一致的問題已解決，這個換算方式定案。
  - **加權後的失衡（`information`/`weight` 公式本身的問題，不是單位問題）**：
    加上 `information = inlier_count * eye(6)`（`inlier_count=40`）跟固定
    `anchor weight=1.0` 之後，加權比例還有 ≈236.6x（2.37 個數量級）——
    這是 `inlier_count` 沒有歸一化造成的，跟像素/公尺單位無關。
  - **`inlier_count_reference` 定案：用這 9 條邊的 median（1861），不是任選
    一條邊的實測值**。用真實資料對 `data/smoke/` 全部 10 張影像的
    `sequential_pairs()` 9 條邊逐一跑過真實 `match_pair`（SIFT+ratio test），
    inlier_count 落在 40～3061（mean=1663.7, median=1861），只有前兩條邊
    （0→1、1→2）是 40，其餘 7 條邊都遠高於 40——用 40 當 reference 不具代表性
    （它是最小值、離群值，不是「正常水準」）；用 mean 也不理想，會被這兩個
    離群值往下拉。9 條邊完整診斷表格（`information = (inlier_count/1861) *
    eye(6)`，`pixels_per_meter≈28.703`，anchor weight=1.0）：

    | edge | inlier_count | tx_px | ty_px | gps_delta_m | raw_edge_residual_px | information (對角線係數) | weighted_edge_residual | raw_anchor_residual_px | weighted_ratio |
    |------|-------------:|------:|------:|------------:|----------------------:|--------------------------:|------------------------:|------------------------:|---------------:|
    | 0→1  | 40   | 2416.6 | -675.9 | 14.78 | 2509.3 | 0.0215 | 53.9    | 424.2 | 0.13 |
    | 1→2  | 40   |  977.8 | -583.4 |  9.69 | 1138.6 | 0.0215 | 24.5    | 278.1 | 0.09 |
    | 2→3  | 717  | 1697.2 | -682.1 | 14.40 | 1829.1 | 0.3853 | 704.8   | 413.3 | 1.71 |
    | 3→4  | 1861 |  -82.6 |  364.2 | 13.28 |  373.4 | 1.0000 | 373.4   | 381.2 | 0.98 |
    | 4→5  | 1853 |  -49.7 |  407.4 | 13.66 |  410.4 | 0.9957 | 408.6   | 392.1 | 1.04 |
    | 5→6  | 2405 |  -75.2 |  333.7 | 13.01 |  342.1 | 1.2923 | 442.2   | 373.4 | 1.18 |
    | 6→7  | 2192 |  -39.3 |  363.3 | 13.53 |  365.4 | 1.1779 | 430.4   | 388.4 | 1.11 |
    | 7→8  | 2804 |  -13.8 |  443.2 | 13.99 |  443.4 | 1.5067 | 668.1   | 401.6 | 1.66 |
    | 8→9  | 3061 |  -44.7 |  431.3 | 12.24 |  433.6 | 1.6448 | 713.2   | 351.3 | 2.03 |

    加權比例橫跨全部 9 條邊落在 **0.09x～2.03x**（-1.06～0.31 個數量級），
    對照組（reference=40）是 4.09x～94.45x（0.61～1.98 個數量級），
    reference=mean(1663.7) 是 0.10x～2.27x（-1.01～0.36 個數量級，被
    離群值拉低，範圍跟 median 接近但理論上不如 median 穩健）。
  - **確認 information 數值本身符合設計意圖**：用 median=1861 當分母後，
    inlier_count 最低的兩條邊（0→1、1→2）算出的 information 分別只有
    0.0215（約是其他 7 條邊 0.385～1.645 的 1/18～1/76）——這是機制正確
    運作的證據：這兩條邊的 pairwise homography 本來就最不可靠，理應在
    最佳化目標裡被大幅降權，讓 GPS anchor 在這裡發揮相對更大的拉力，
    不是需要修正的異常。
  - **最終定案**：`information = (inlier_count / 1861) * eye(6)`
    （`inlier_count_reference = 1861`，這批 `data/smoke/` 9 條邊實測
    inlier_count 的 median），取代先前用單一邊 `inlier_count=40` 當
    reference 的版本（40 是離群值、不具代表性，已被這次完整 9 邊驗證
    推翻）。`pixels_per_meter≈28.703` 的換算方式維持不變。

## 目前狀態
- [x] SSH + VS Code Remote-SSH + Claude Code CLI 環境
- [x] 專案骨架
- [x] metrics.py + unit tests
- [x] EXIF/XMP 解析 (GPS 座標讀取 + 局部平面投影，21/21 tests passing)
- [ ] direct georeferencing (geo/camera.py, geo/direct.py) 仍暫緩，見「已知的暫緩事項」
- [ ] feature-based pipeline
  - [x] estimate.py: sequential_pairs + match_pair/estimate_all_pairs
    (SIFT + RANSAC, 43/43 tests passing)
  - [x] posegraph.py: optimize_pose_graph (Sim(2) 最小二乘 + GPS anchor，
    45/45 tests passing，另用不對稱合成資料驗證過 node index 對應正確)
  - [x] posegraph.py: build_pose_graph (從真實 PairResult + GPS 座標建圖)
  - [x] compose.py: compose_global_transforms
  - [ ] warp.py
  - [ ] blend.py
  - [ ] pipeline.py: 串接 estimate → compose → warp → blend
- [ ] FastAPI
