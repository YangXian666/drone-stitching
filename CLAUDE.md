

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
- **這個 pod 的真實記憶體上限是 60GB（cgroup `/sys/fs/cgroup/memory.max`），
  不是 `free -h` 顯示的主機總量（314GB）**——`free -h` 看到的是整台共用
  主機的記憶體，跟這個 pod 實際能用的量無關，會嚴重誤導記憶體相關的判斷。
  任何記憶體診斷或容量規劃都要看 `/sys/fs/cgroup/memory.current`（目前
  用量）對照 `/sys/fs/cgroup/memory.max`（上限，這個 pod 是 60GB），不要
  再看 `free -h`。已經用一次真實診斷（`warp_images`/`blend_images` 對
  20~50 張真實影像的記憶體用量，見下面「已知的限制」）驗證過這個落差：
  `free -h` 顯示「還有 266Gi 可用」的當下，這個 pod 實際離 60GB 的 cgroup
  上限只剩不到 10GB
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
- **⚠️ 2026-09-26 更新：下面這條「暫緩到 10 月」的決定已被取代**——後來發現
  GPS anchor 框架有一個鏡射 bug，解釋了這條診斷鏈的大部分現象，決定現在就改用
  分階段架構（Stage A～D）處理。見「已知的限制」的「GPS anchor 框架鏡射 bug 與
  分階段架構決定（2026-09-26）」。下面原文保留，作為當時推理的紀錄。
- **`compose_global_transforms` 的旋轉退化問題（見下面「已知的限制」的完整
  診斷鏈：GPS anchor 耦合、decoupled 公式仍耦合、純旋轉鏈本身複合退化）
  ——暫緩，等 10 月正式資料集重新評估，不是現在要修的 bug**。理由：
  用合成資料的測試已經證明 `optimize_pose_graph`/`YawAnchor` 本身的核心
  邏輯是正確的（27+ 條合成資料測試全綠，`YawAnchor` 在乾淨的合成情境下
  確實能救援退化），問題根源不是架構寫錯了，是**這批網路取得的替代資料
  本身，homography 估計精度不足以支撐純幾何聯合最佳化**——這正是這批
  資料一開始就被選作 smoke test（而非正式資料）的已知限制（見上一條：
  相機規格靠推算、metadata 格式不可靠）。三個獨立機制（GPS anchor 透過
  `inv()` 耦合、decoupled 公式仍透過 `R_dst @ t_rel` 耦合、純旋轉鏈本身
  在缺乏外部錨定時複合退化）連續指向同一個結論：`GimbalYawDegree` 是
  這批資料裡唯一穩定的旋轉訊號來源。兩個候選解法（兩階段求解、或旋轉
  直接採用 `GimbalYawDegree` 常數不參與最佳化）都先不動手——**如果 10
  月的正式資料集換了更適合海面的 matcher、或影像重疊率更高、homography
  估計精度明顯提升，這個問題可能自然消失，不需要現在就投入複雜的架構
  重構去解決一個可能是這批資料特定的問題**。拿到正式資料後，先重新跑一次
  「純旋轉鏈複合退化」的診斷（見下面「已知的限制」的 Check A），如果
  問題還在，再決定要不要實作這兩個候選解法之一。
- **在這個問題解決之前，`compose_global_transforms` 的輸出在旋轉分量上
  不可靠——`warp.py`/`blend.py` 開發時要明確處理這個限制**：下游先只用
  平移座標做粗略的畫面排列驗證（例如確認影像大致排在正確的相對位置），
  不依賴精確的旋轉結果去做真正的透視變形/拼接；`warp.py` 的正式
  correctness 驗證應該留到旋轉退化問題有解（或確認 10 月資料集沒有這個
  問題）之後再做，不要在已知旋轉不可靠的狀態下就當作 `warp.py` 的品質
  基準。

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
- **`compose_global_transforms` 的旋轉/縮放子空間會嚴重退化（scale 崩潰、
  旋轉不連續甚至變號），根因有兩層，已確定修法方向是新增 `YawAnchor`，
  但 `YawAnchor` 本身還沒實作（見下面「目前狀態」）**。用 `data/smoke/`
  全部 10 張真實影像、9 條真實邊（真的 SIFT+ratio-test matcher，不是
  合成資料）做過完整診斷，過程與結論如下：
  - **症狀**：GPS-anchored 版本裡，node 1、2、3 的 scale（`sqrt(a²+b²)`）
    分別是 0.169、0.244、0.226（遠低於 0.8 的合理下限，等於把影像壓縮
    成不到 1/4 大小），旋轉角度完全不連續甚至變號（-59.4°→-13.7°→
    +28.0°），跟真實 `GimbalYawDegree` 反映的偏航變化趨勢（0°→-62.2°→
    -93.6°→-143.3°）完全對不上。這個狀態如果不修，`warp.py` 會直接
    顯形成扭曲影像，不能帶著往下走。
  - **對照實驗排除了「融合有效、只是剛好接近 GPS」的可能性**：拿同一批
    9 條真實邊跑 `compose_global_transforms(gps_positions=None, ...)`
    （純 edge 約束，不用 GPS anchor），結果跟 GPS-anchored 版本的座標
    差距最大到 194m；但 edge-only 版本的旋轉角度（-61.6°→-92.6°→
    -140.9°，之後持平）反而跟真實 `GimbalYawDegree` 高度吻合——證明
    homography 的旋轉分量本身是可信的物理訊號，問題出在 GPS-anchored
    版本裡這個訊號被結構性壓制，不是訊號本身有問題。
  - **根因 1**：`GPSAnchor` 的殘差公式只用 `pose[:2,2]`（tx,ty），從來
    不碰 `pose[:2,0:2]`（a,b，即旋轉/縮放）。當強力的平移 anchor 把某個
    node 的 tx,ty 拉向一個跟 homography 隱含平移不一致的位置時，唯一能
    吸收這個矛盾的自由度就是旋轉/縮放，而這個問題在低 inlier 邊（0→1、
    1→2，inlier_count=40，information 被 median 正規化壓到只剩 0.0215，
    是其他邊的 1/18～1/76）相鄰的 node 上最嚴重，因為那裡幾乎沒有任何
    有效約束在管旋轉/縮放。
  - **根因 2（比根因 1 更根本，獨立於 GPS anchor 存不存在）**：
    `information = (inlier_count/inlier_count_reference) * eye(6)` 這個
    設計，對 6 維殘差向量 `(predicted - relative_pose)[:2,:].flatten()`
    裡的旋轉分量（index `{0,1,3,4}`，無因次矩陣元素，量級 0.02～1.5）和
    平移分量（index `{2,5}`，像素單位，量級 500～2500）套用同一個純量
    係數。用真實 9 條邊在初始猜測狀態下實測：`full_raw`/`full_wtd`
    幾乎完全等於 `trans_raw`/`trans_wtd`（例如 0→1 邊 `full_raw=2322.524`
    vs `trans_raw=2322.523`），旋轉子區塊（`rot_wtd`，範圍 0.0164～
    0.4615）在混合 norm 裡幾乎不可見。也就是說**不管 information 係數
    設多少，edge 殘差對旋轉分量施加的有效壓力永遠遠低於對平移分量的壓力
    （相差 3～4 個數量級）**——這是一個獨立於 GPS anchor 之外，
    `optimize_pose_graph` 核心殘差公式本身就有的單位失衡問題。**這個
    問題這次先記錄、不處理**（不在 `YawAnchor` 這次順手改動核心殘差
    邏輯），但已列為獨立的架構待辦（見下面「目前狀態」），因為它比
    `YawAnchor` 的 weight 調校更根本，屬於另一類問題。
  - **`YawAnchor` 設計定案（尚未實作）**：
    - 資料流分工：`io_utils.load_gimbal_yaw`（已實作，63/63 tests
      passing）只讀 raw `GimbalYawDegree`（XMP-only，無 EXIF 對應項）→
      之後在 `geo/projection.py` 新增一個跟 `project_gps_positions` 對稱
      的 `project_gimbal_yaw_degrees`（純粹算相對 origin 的 wrap 角度，
      不知道 Sim(2)/homography）→ 符號翻轉（`H_angle ≈ -relative_yaw`，
      驗證見下方）+ 轉單位向量的邏輯放在 `posegraph.py`（pose-graph
      專屬知識，不外露到 `geo/projection.py`）。
    - `YawAnchor` dataclass 跟 `GPSAnchor` 平行（`image_index`、
      `target_vector`，即 `[cos θ_target, sin θ_target]`、`weight`），
      不做成通用「殘差類型」抽象——目前只有 2 種 anchor，還不到需要
      抽象化的規模（Rule of Three），先讓兩個 dataclass 形狀一致，方便
      以後真的出現第三種時再升級成共用介面。`PoseGraph` 新增
      `yaw_anchors: list[YawAnchor]` 欄位（給 `default_factory=list`，
      不破壞現有直接建構 `PoseGraph(...)` 的呼叫點）。殘差公式
      `pose[:2,0] - anchor.target_vector`，隱含約束 `scale≈1`（這批
      影像飛行高度只變動 <0.1%，`scale≈1` 本來就該成立，不用另外設計
      scale 欄位）。
    - `build_pose_graph` 新增 `gimbal_yaw: dict[int, float] | None = None`
      （跟現有 `gps_positions` 一樣可選——沒有 yaw 資料時退回目前狀態，
      仍是合法的 pose graph）+ `yaw_anchor_weight: float | None = None`
      （必填但用執行期檢查而非型別系統強制：`gimbal_yaw` 給了、
      `yaw_anchor_weight` 卻是 `None` 要 raise，不能靜默套用未驗證的
      預設值，跟 `pixels_per_meter`/`inlier_count_reference` 同樣的
      原則——量級沒驗證過就不能預設）。
  - **符號翻轉關係驗證（9 條邊全數驗證，不是只挑吻合的兩條）**：
    `residual = H_angle - (-relative_yaw)` 在 9 條邊上落在 -1.52°～
    +0.67°（mean≈-0.30°，std≈0.68°），且**最關鍵的發現**是兩條低 inlier
    的邊（0→1、1→2，inlier_count=40）的殘差（-0.124°、+0.029°）是全部
    9 條邊裡最小的兩個，反而比某些高 inlier 邊（2→3，717 inlier，殘差
    -1.519°）更準——證明這兩條邊的旋轉分量本身完全可信，只是被
    information 正規化壓到失聲，`YawAnchor` 能救回一個結構性被消音、
    但本身是對的訊號。**樣本限制附帶條件**：這 9 條邊只來自單一飛行
    高度、單一直線飛行序列，9 條裡只有 3 條有實際偏航變化，尚未涵蓋
    轉彎/爬升等更複雜飛行動作——10 月拿到正式資料集後，如果飛行模式
    更複雜，這個符號翻轉關係要重新驗證，不能直接沿用。
  - **weight 量級診斷（用跟 `pixels_per_meter`/`inlier_count_reference`
    同樣的方式，不能憑感覺選係數）**：在初始猜測狀態下比較，yaw anchor
    的原始殘差（node1~9 分別是 1.033、1.458、之後持平 1.898）要跟 edge
    殘差的「旋轉子區塊」`rot_wtd`（9 條邊範圍 0.0164～0.4615，中位數
    0.0351）比，不是跟完整混合殘差 `full_wtd`（569.08，會被平移污染，
    算出來的 weight≈300 會讓 `YawAnchor` 重演 GPS anchor 那種壓倒性
    主導）。用 `rot_wtd` 中位數算出 `weight ≈ 0.0185`，建議起始值
    **`weight≈0.02`，範圍 0.01～0.05**。
- **`YawAnchor` 落地後，實質上會是旋轉分量的主要、甚至唯一有效約束
  來源，不是單純的「補強」——這是根因 2（information 單位失衡）的直接
  後果，是一個需要明確記錄的依賴風險，不只是附帶條件**：因為 edge
  殘差對旋轉分量的約束力天生就遠弱於平移分量（見上面根因 2），加上
  `YawAnchor` 之後，旋轉分量的可靠性幾乎完全取決於 `GimbalYawDegree`
  這個 DJI 專屬 XMP 欄位存不存在、準不準。**如果未來某批影像（例如換了
  非 DJI 機型，或 10 月正式資料集的 metadata 格式不同）沒有這個欄位，
  `gimbal_yaw=None`，旋轉分量會退回到現在這個幾乎沒有任何有效約束的
  狀態，而且目前的設計不會有任何警告或降級機制提示這件事發生了**。
  已在下面「目前狀態」列成一個獨立的架構待辦。
  **`weight≈0.02` 這個設計已知的能力邊界**：這個量級是用真實邊的
  info-weighted「旋轉子區塊」殘差（`rot_wtd`，範圍 0.0164～0.4615）
  校準出來的，對應的是「輕微的旋轉分歧」——例如這批真實資料裡兩條低
  inlier 邊（0→1、1→2）的情況：旋轉本身其實是準的，只是被 information
  正規化壓到失聲。**這個 weight 救不回「旋轉本身嚴重錯誤」的情況**
  （不管 inlier_count 高低都可能發生，因為已經驗證過兩者不相關，見
  上面符號翻轉驗證的第 3 點）：用合成資料實測，同樣的嚴重旋轉錯誤（差
  130°）在 `weight=5.0` 才能被拉回（誤差降到 ~3°），但在 `weight=0.02`
  幾乎完全拉不動（結果跟完全沒有 yaw anchor 幾乎一樣，差距只有
  ~0.01°）——這個能力邊界已經用測試鎖定
  （`test_optimize_pose_graph_production_yaw_weight_cannot_rescue_severe_rotation_error`），
  不是現在要解決的問題，只是要明確記錄：**`YawAnchor` 不是萬能的旋轉
  修正機制，只能處理「訊號本身是對的、只是被結構性消音」這一類問題**，
  遇到真正嚴重的旋轉錯誤時，跟完全沒有 `YawAnchor` 的狀態沒有實質差別。
- **`_yaw_target_vector` 曾經有一個符號 bug，已修正——這個 bug 本身、
  它為什麼發生、以及為什麼自動化測試沒攔到它，比修正的公式本身更值得
  記錄**：
  - **錯的公式**：`theta_target = -relative_yaw_deg`。**對的公式**：
    `theta_target = relative_yaw_deg`（不取負號）。
  - **為什麼會犯這個錯**：CLAUDE.md 已經驗證過 `H_angle ≈ -relative_yaw`
    這個關係（見上面符號翻轉驗證的段落），這件事本身沒有錯——但
    `H_angle` 是**edge 的 `relative_pose`**（`inv(pose_dst) @ pose_src`）
    分解出來的角度，不是 `YawAnchor` 實際要約束的**node 絕對姿態角度**。
    從 edge 的角度換算到 node 的角度，中間還有一次矩陣求逆：
    `pose_dst = pose_src @ inv(relative_pose)`，對純旋轉而言，求逆會把
    角度再變號一次：`node_angle = -H_angle = -(-relative_yaw) =
    +relative_yaw`——兩次變號互相抵銷。設計 `_yaw_target_vector` 時漏掉
    了「node 角度 = -H_angle」這一步中間的求逆變號，把 edge 層級驗證過
    的符號關係直接套用到 node 層級的目標公式上，等於少變號一次。
  - **為什麼測試沒攔到**：`test_yaw_target_vector_applies_validated_sign_flip`
    當初的期望值是用同一個（錯的）推導手算出來的——期望值和實作犯了
    同一個錯，兩者自然吻合、測試綠燈，但驗證的是「實作是否符合我自己
    （錯誤）的理解」，不是「實作是否符合真實物理」。其餘會用到
    `_yaw_target_vector` 的測試（`build_pose_graph` 相關）在比對期望值時
    也是直接呼叫 `_yaw_target_vector` 本身去算期望值（`anchor.target_vector
    == pytest.approx(_yaw_target_vector(relative_yaw_deg))`），這種「用
    同一個函式算期望值」的寫法對任何函式內部的系統性錯誤都是免疫的、
    抓不到。真正涉及旋轉數值的整合測試（`optimize_pose_graph` 的隔離
    拉力測試、嚴重退化救援測試）則是直接用 `cos`/`sin` 手算
    `target_vector`，完全不經過 `_yaw_target_vector`，所以也没機會踩到
    這個 bug——這些測試驗證的是「`optimize_pose_graph` 的殘差公式接線
    正確」，從設計上就沒有涵蓋「`relative_yaw` 轉 `target_vector` 這個
    轉換方向對不對」這件事。
  - **後來怎麼抓到的**：不是靠測試，是靠**用真實 `data/smoke/` 資料重跑
    一次「拉不動」的 weight sweep 時發現異常**——把 `weight` 從 0.02
    拉到 10.0，node1 的角度幾乎紋風不動，而且用（當時還是錯的）目標角度
    去算「node 4~9 離目標差多少」時，跑出 228°~238° 這種明顯不合理的
    數字，回頭重新推導才抓到。這組「拉不動」的 weight sweep 數字因此
    整批作廢，需要修正符號後重新跑一次才可信。
  - **這件事證明了什麼**：這個 session 一路堅持的「不能只看合成測試綠燈
    就當作完成，要用真實資料驗證」這個方法論最終確實有效——真的抓到了
    一個合成測試設計上結構性看不見的 bug。但也要誠實承認：這次是靠**用
    真實資料的最終行為（拉不動）反推出數字不合理**才抓到的，不是靠更
    嚴謹的測試設計主動攔截的——如果當初有一條測試是「用 `load_gimbal_yaw`
    +`project_gimbal_yaw_degrees` 的真實輸出餵給 `_yaw_target_vector`，
    再檢查算出來的角度是否讓 edge-only（無 anchor）的已知收斂結果得到
    低殘差」，這類**串接真實資料端到端、且期望值來自獨立驗證過的其他
    資料（不是同一個函式自我比對）**的測試，理論上應該要能在 TDD 階段
    就攔到這個 bug，而不必等到用生產 weight 在真實資料上跑出異常才發現。
- **修正符號 bug 後重新驗證：`YawAnchor` 在真實資料上，`weight` 從
  0.02 調到 10.0 都救不回退化，而且發現退化其實影響全部 9 個 node，
  不只是原本判定的 node1/2/3——這組發現最終指向 `optimize_pose_graph`
  的殘差公式本身有一個經典的 Sim(2) pose-graph 耦合陷阱，需要重新設計，
  但這次刻意不動手實作，完整記錄在這裡，留給下一個 session 處理**：
  - **修正符號後的 weight sweep 結果（真實 9 條邊，`weight` ∈
    {0.02, 0.5, 1.0, 2.0, 5.0, 10.0}）**：`residual_error` 隨 weight
    增加持續惡化（0.2020 → 2.8890，漲了 14 倍），但 node1/2/3 的角度
    誤差幾乎不動（node1 穩定在 2.77°——其實從一開始就沒有嚴重偏離，
    真正的問題是 scale 卡在 0.169～0.170 動不了；node2 誤差 79.9°→
    77.6°，只改善 2.3°；node3 誤差 171.3°，完全不動）。**意外發現**：
    node4～9（先前因為 scale 維持在 0.81～1.09 而被判定「健康」）的
    旋轉角度其實一直跟真實 `GimbalYawDegree` 差 47.8°～58.6°，這個
    誤差在整個 weight sweep 裡也幾乎不動——「只看 scale 判斷健康與否」
    是不完整的判準。
  - **三步排查，排除了「又是一個符號 bug」的可能性**：(1) 用真實數字
    驗證 `project_gimbal_yaw_degrees[9]` 精確等於
    `GimbalYawDegree[9]-GimbalYawDegree[0]`（wrap 後），10/10 個 node
    全部吻合，函式本身沒問題；(2) 純 edge 鏈（無 GPS anchor）本身的
    旋轉一路都準（node9 只偏了 6.58°，屬於 9 段真實雜訊合理累積量級），
    證明目標角度公式是對的，偏差不是計算錯誤；(3) 拿純 edge 鏈的實際
    收斂角度對照「逐邊 `H_angle` 手動累加」的天真預測，差距隨鏈長平緩
    遞增到 3.9°（3→9 node），量級跟真實雜訊累積一致，不是被漏轉一次
    符號那種量級（那種 bug 差距應該接近 60°或 120°這種數量級，不會是
    平滑遞增的個位數度數）。**結論：目標角度計算本身沒有第二個 bug，
    GPS anchor 一旦加入，確確實實會把整條鏈的旋轉拉偏 48°～171°不等，
    是一個真實現象。**
  - **逐邊隔離實驗：證明退化是每條邊獨立發生，不是從 node1 傳染過來**。
    對 9 條真實邊逐一做「只留 src、dst 兩個 node + 各自 GPS anchor +
    這一條邊，完全不接其他 node」的隔離實驗（`reference_index=src`），
    結果：

    | edge | info_coef | 隔離後 scale | 隔離後相對角度 | 真實相對偏航 | 隔離誤差 |
    |------|----------:|-------------:|---------------:|-------------:|---------:|
    | 0→1 | 0.0215 | 0.1690 | -59.435° | -62.200° | 2.765° |
    | 1→2 | 0.0215 | 0.2443 | -13.692° | -31.400° | 17.708° |
    | 2→3 | 0.3853 | 0.2259 | 28.003° | -49.700° | 77.703° |
    | 3→4 | 1.0000 | 1.0204 | -94.826° | 0.000° | 94.826° |
    | 4→5 | 0.9957 | 0.9553 | -89.980° | 0.000° | 89.980° |
    | 5→6 | 1.2923 | 1.0919 | -95.506° | 0.000° | 95.506° |
    | 6→7 | 1.1779 | 1.0624 | -89.137° | 0.000° | 89.137° |
    | 7→8 | 1.5067 | 0.9056 | -84.746° | 0.000° | 84.746° |
    | 8→9 | 1.6448 | 0.8105 | -88.880° | 0.000° | 88.880° |

    **每一條邊單獨拿出來都出現嚴重退化**，包括在全鏈裡看起來「健康」的
    3→4 到 8→9（單獨拿出來角度誤差高達 84.7°～95.5°，一點都不健康，
    只是在全鏈裡因為彼此誤差方向接近、相對誤差小，被掩蓋成「看起來
    健康」）。這推翻了「node1 是病灶、往後傳染」的假設，確認是「每一條
    邊只要 GPS anchor 隱含位置跟 homography 隱含位置不一致，(a,b) 就會
    獨立退化去吸收這個矛盾」——`information` 高低、node 在鏈上的位置、
    上游有沒有先出問題，都不影響這個現象是否發生。
  - **根因（Sim(2) pose-graph 的經典耦合陷阱）**：`optimize_pose_graph`
    目前用 `predicted = inv(poses[dst]) @ poses[src]` 算完整個矩陣再跟
    `relative_pose` 整包相減。對 Sim(2) 矩陣求逆會產生 `1/(scale²)` 這種
    項，讓平移殘差的數值透過矩陣乘法「污染」進旋轉/縮放子空間——當
    GPS anchor 把 `t_dst` 拉向一個跟 edge 隱含位置不一致的值時，
    最小平方法發現讓 `(a,b)`（旋轉/縮放）退化到一個極端值，比讓
    `t_dst` 動更「便宜」（見上面「已知的限制」中 information-vs-退化
    幅度診斷：`information` 係數在 6 個數量級範圍內幾乎不影響退化程度，
    證明問題不是權重，是殘差公式本身的耦合結構）。
  - **修法：把殘差拆成獨立的旋轉子項和平移子項，不用 `atan2`（不影響
    wraparound 顧慮），全程不對決策變數取逆**。從
    `relative_pose ≈ inv(pose_dst) @ pose_src` 兩邊左乘 `pose_dst`：
    `pose_dst @ relative_pose ≈ pose_src`。拆開旋轉/平移分量：
    - 旋轉殘差（4 個數）：`R_dst @ R_rel − R_src`，其中
      `R_rel = relative_pose[:2,:2]`（edge 資料本身的固定值，不是決策
      變數，取它不需要求逆決策變數）。
    - 平移殘差（2 個數）：`(t_dst + R_dst @ t_rel) − t_src`，其中
      `t_rel = relative_pose[:2,2]`（同樣是固定資料）。

    兩者都是決策變數（`R_src, R_dst, t_src, t_dst`）的線性/雙線性組合，
    全程沒有對任何決策變數取逆——GPS anchor 對 `t_dst` 的拉力，只會
    透過乘法線性傳到平移殘差，不會再透過 `inv()` 的 `1/(scale²)` 項滲進
    旋轉子空間。`PairResult`、`PoseGraphEdge`、`PoseGraph`、`GPSAnchor`、
    `YawAnchor` 這幾個 dataclass 的欄位/型別都不需要改；`edge.information`
    還是同一個 6x6 矩陣，套用在重新組回的 6 維 `[旋轉殘差, 平移殘差]`
    向量上，跟現在的寫法完全對稱。改動範圍侷限在
    `optimize_pose_graph` 內部 `residuals()` closure 處理 `graph.edges`
    的那個迴圈本體（約 10-15 行），函式簽名、資料結構全部不變。
  - **範圍評估的三點結論**（詳細分類見對話紀錄，這裡記結論）：
    (1) `tests/test_posegraph.py` 裡約 19 條測試（`build_pose_graph`
    相關、`YawAnchor`/`_yaw_target_vector` 相關）完全不受影響，因為測的
    是 `build_pose_graph` 產出的資料結構內容，不涉及
    `optimize_pose_graph.residuals()` 內部算法；(2) 約 6 條測試
    （`optimize_pose_graph`/`compose_global_transforms` 的收斂測試、
    GPS anchor 救援測試、YawAnchor 旋轉退化救援測試）需要重跑確認仍然
    通過，但這些測試全部是門檻式斷言（`error < X`、`< 0.3 * 其他值`），
    不是手推精確數字，且合成真值多半旋轉為單位矩陣（舊公式耦合機制
    影響最小的情況），預期會通過但不能只憑推論假設；(3) **0 條測試
    需要重新手推期望值**。
  - **⚠️ 明確標註：以下這些數字全部是在舊（有耦合陷阱的）殘差公式下
    跑出來的，新公式實作完成後必須重新驗證是否還成立，不能沿用**：
    - `inlier_count_reference = 1861`（median 校準）——這個數字本身
      （這批 9 條邊 `inlier_count` 的 median）不會變，但它「能讓
      information 發揮設計意圖」這個結論是在舊殘差公式下驗證的，新
      公式下 information 係數影響退化程度的方式可能完全不同，需要
      重新驗證。
    - `YawAnchor` 的 `weight≈0.02`（範圍 0.01～0.05）建議值——完全是
      在舊殘差公式下校準與驗證的（包含 `rot_wtd` 量級比較、合成資料的
      weight=5.0 才能救援嚴重錯誤等結論），新公式解決了根本的耦合問題
      後，`YawAnchor` 還需不需要、需要多大的 weight，都要重新評估——
      新公式甚至可能讓 `YawAnchor` 變得不必要（如果 GPS anchor 不再
      污染旋轉子空間，node 的旋轉可能直接由 edge 自己撐住）。
    - **不受影響的部分**：`pixels_per_meter≈28.703` 這個換算方式**不受
      影響**——它是純幾何推導（DFOV + 飛行高度算地面覆蓋），不依賴
      `optimize_pose_graph` 的任何殘差計算，新公式上線後不需要重新
      驗證這個數字。
- **上面那個「解耦殘差公式」已經實作完成、88/88 測試全綠——但用真實
  `data/smoke/` 資料重新驗證後發現：它沒有解決原始的 scale/旋轉退化
  問題。這是這整條除錯鏈裡最重要的一次結論修正，必須完整記錄**：
  - **新公式本身的推導（已實作，`旋轉殘差`部分確實完全乾淨）**：從
    `relative_pose ≈ inv(pose_dst) @ pose_src` 兩邊左乘 `pose_dst`：
    `pose_dst @ relative_pose ≈ pose_src`。拆開後：
    - 旋轉殘差（4 個數）：`R_dst @ R_rel − R_src`——**這部分完全不涉及
      任何平移量**（`t_dst`、`t_src` 都不出現），是真正乾淨的旋轉子問題。
    - 平移殘差（2 個數）：`(t_dst + R_dst @ t_rel) − t_src`。
    全程不對決策變數取逆（`R_rel = relative_pose[:2,:2]`、
    `t_rel = relative_pose[:2,2]` 都是 edge 資料本身的固定值）。已實作
    在 `optimize_pose_graph` 的 `residuals()` closure 裡（取代原本
    `predicted = inv(poses[dst]) @ poses[src]` 那行），`PairResult`、
    `PoseGraphEdge`、`PoseGraph`、`GPSAnchor`、`YawAnchor` 的欄位/型別
    都沒有改動，函式簽名不變。
  - **真實資料驗證結果：沒有改善**。重跑最初的診斷（`data/smoke/` 9 條
    真實邊，不加 `YawAnchor`），node1/2/3 的 scale/角度**跟修正前幾乎
    一模一樣**（0.1690/-59.435°、0.2442/-13.664°、0.2259/28.002°，
    小數點後 3-4 位都相同）。
  - **用矩陣代數精確推導出兩個公式的關係，解釋為什麼「幾乎一模一樣」
    不是巧合**：`old_diff = -inv(T_dst) @ new_diff`（推導：
    `inv(T_dst)@T_src - Rel = inv(T_dst)@(T_src - T_dst@Rel) =
    -inv(T_dst)@(T_dst@Rel - T_src) = -inv(T_dst)@new_diff`，已用真實
    矩陣數值驗證過這個恆等式精確成立）。因為 `inv(T_dst)` 只有在
    `T_dst` 的 scale 恰好等於 1 時才是正交矩陣（範數不變），一開始用
    scale=1 的測試點驗證「兩個公式範數相等」得到了誤導性的巧合結果；
    改用 scale≈1.34（實際收斂值的量級）的測試點重新驗證，範數比例精確
    等於 `1/scale`（不是 1）——證明兩個公式**在數學上真的不同**。但
    用 `scipy.least_squares` 提高到 `xtol=ftol=gtol=1e-15` 高精度重解，
    兩者的最佳解只在小數點後 4～7 位不同（例如 `info_coef=5.0` 時最大
    參數差距 `7.29e-03`，相對參數量級 ~1000 幾乎可忽略）——**耦合機制
    確實從「`1/scale²` 反比關係」換成了「`R_dst @ t_rel` 線性乘法關係」，
    強度有減弱，但沒有真正切斷，實務上幾乎不影響最佳化找到的解**。
  - **關鍵判斷（比公式本身更重要）**：`R_dst`（決策變數）出現在平移
    殘差 `(t_dst + R_dst @ t_rel) − t_src` 裡，**這很可能是幾何上必然
    的，不是這次公式選錯了**——要把 edge 自己座標系裡量出來的平移向量
    `t_rel`，轉換到跟 `t_src`/`t_dst` 同一個全域座標系下比較，本來就
    需要用 `t_dst` 所在 node 的旋轉把 `t_rel` 轉過去，這個旋轉沒有辦法
    在保留「平移殘差」這個概念的前提下被拿掉。也就是說：**問題可能不在
    殘差公式怎麼寫（旋轉/平移拆不拆、對誰求逆），而在「用單一聯合最小
    二乘法同時求解旋轉和平移」這個框架本身有結構性限制**——只要旋轉和
    平移在同一個最佳化問題裡同時是自由變數，边的平移殘差就沒有辦法
    完全不依賴旋轉，GPS anchor 對平移的拉力就永遠有機會透過這個依賴
    關係去扭曲旋轉。
  - **下一個方向（尚未評估細節、尚未動手）：兩階段求解（Rotation
    Averaging 再 Translation）**，這是攝影測量/SLAM 領域處理這類耦合
    問題的標準做法：
    - **第一階段（Rotation Averaging）**：只用旋轉殘差
      `R_dst @ R_rel − R_src`（已經完全乾淨，不涉及任何平移量）+
      `YawAnchor` 的殘差（本來就是純旋轉約束）求解全部 node 的旋轉
      （含 scale，因為這批 Sim(2) 的 `(a,b)` 本來就同時編碼兩者），
      完全不碰任何平移變數。`GPSAnchor` 在這個階段完全不參與（它本來
      就設計成只管平移，這正好吻合）。
    - **第二階段（Translation）**：把第一階段解出來的旋轉當成**固定
      資料**（不再是決策變數），平移殘差 `(t_dst + R_dst @ t_rel) −
      t_src` 裡的 `R_dst @ t_rel` 這一項此時已經是常數，整個平移殘差
      變成對 `t_src`、`t_dst` 的**純線性**組合——`GPSAnchor` 的殘差（本來
      就是純平移）也在這個階段一起求解。
    這個方向目前只記錄構想，**還沒有評估兩階段求解對 `optimize_pose_graph`
    介面、既有 27 條測試、以及整體工作量的具體影響——這是下一步要做的
    範圍評估，還沒有做**。
  - **範圍評估完成後，動手實作前先驗證了範圍評估自己提出的疑點（第一
    階段「純旋轉最小平方法」本身會不會有自己的退化模式）——結果發現
    一個新的、獨立的退化機制，改變了整個方向，兩階段設計最終被放棄**：
    用一個獨立診斷腳本（沒有進 `posegraph.py`，純粹驗證用），只取旋轉
    殘差 `R_dst @ R_rel − R_src` 和 `YawAnchor` 殘差，對真實 9 條邊 +
    `GimbalYawDegree` 資料跑純旋轉 `least_squares`（完全不涉及任何平移
    變數），做了三項檢查：
    - **Check A（9 條邊鏈，無 `YawAnchor`）：scale 嚴重崩潰**。角度幾乎
      完全準確（跟先前驗證過的 edge-only 結果一致，node1~9 誤差
      0.6°~6.6°），但 **scale 崩潰到 0.13～0.57**，連 information 最高
      的邊（3→4 到 8→9，info_coef 1.0～1.6）撐起來的 node 也一樣崩潰
      到 ~0.13。**這是一個全新、獨立的退化機制**：多條邊各自帶有的
      小幅不完美（真實 homography 的 2x2 旋轉子矩陣不是完美的相似變換，
      單邊 `H_scale` 偏離 1 約 5～10%），在完全沒有外部錨定的情況下
      聯合最佳化求解時複合放大——**跟 information 高低無關**（高
      information 的邊一樣崩潰），這是這次除錯鏈裡第三個「information
      不是問題所在」的獨立證據。
    - **Check C（單一邊隔離）：乾淨，證明退化來自鏈式複合，不是單邊
      本身**。0→1 邊單獨解出 scale=0.9371（誤差 0.616°），3→4 邊單獨
      解出 scale=0.9857（誤差 0.824°）——都在合理範圍，沒有病態解。
      證明 Check A 的嚴重崩潰是「多條各自輕微不完美的邊鏈接在一起聯合
      求解」才會出現的複合效應，不是任何單一邊造成的。
    - **Check B（加上 `YawAnchor`）：確實有效，但 `weight≈0.02` 遠遠
      不夠**。`weight=0.02`（目前生產值）時 scale 還在 0.52～0.81 之間
      亂跑（`max|scale-1|=0.4769`）；`weight=1.0` 時全部 10 個 node 的
      scale 收斂在 0.98～1.02、角度誤差全部 <1.1°——乾淨、健康的結果。
      **這代表先前校準的 `weight≈0.02` 建立在錯誤的基準上**：那是用
      舊（耦合）公式在「初始猜測狀態」下的 `rot_wtd` 校準的，跟「乾淨
      的純旋轉鏈本身需要多少拉力才能對抗多邊複合退化」是兩件不同的事，
      實際需要的量級差了將近 50 倍（`0.02` → `~1.0`）。
    - **一個需要特別強調的新風險**：純旋轉 Stage 1 如果失去
      `GimbalYawDegree`（`gimbal_yaw=None`），退化範圍比現有的聯合公式
      **更廣、更嚴重**——現有聯合公式至少讓 node4~9 的 scale 維持在
      0.81~1.09（平移殘差的巨大量級某種程度上稀釋了純旋轉部分的複合
      效應），但純旋轉 Stage 1 一旦失去 `YawAnchor`，全部 10 個 node
      都會崩潰到 0.13~0.57。**兩階段設計會讓整條 pipeline 更依賴
      `GimbalYawDegree`，不是更不依賴**，這跟上面「`YawAnchor` 依賴
      風險」的既有待辦是同一類問題，但風險程度更高。
  - **關鍵的元結論——三個獨立機制連續指向同一個答案**：這條除錯鏈已經
    連續發現三個表面上不同、但根因同構的機制：(1) GPS anchor 的平移
    拉力透過 `inv()` 的 `1/scale²` 污染旋轉子空間；(2) 把公式改成
    decoupled 版本後，`R_dst` 仍然透過 `R_dst @ t_rel`（線性形式）耦合
    進平移殘差，換湯不換藥；(3) 就算把旋轉徹底獨立成一個子問題，純
    旋轉鏈本身在缺乏外部錨定時也會複合退化。**這三次獨立驗證都指向
    同一個結論：這批 `data/smoke/` 資料的 homography 估計精度，天生
    不足以支撐純幾何聯合最佳化（不管是聯合求解、decoupled 聯合求解、
    還是兩階段分開求解），`GimbalYawDegree` 是這批資料裡唯一穩定、
    獨立、可信的旋轉訊號來源，不是「錦上添花的加分項」**。
  - **改變方向：放棄兩階段最小平方法，改用更直接的做法——有
    `GimbalYawDegree` 時旋轉直接採用感測器讀數（不進最佳化，當作已知
    常數），最佳化只求解平移**。具體構想：`gimbal_yaw` 有提供時，
    每個 node 的旋轉直接用 `project_gimbal_yaw_degrees` 算出的值（透過
    `_yaw_target_vector` 轉成 `(a,b)`），完全不當決策變數；最佳化只解
    `(tx,ty)`，殘差用 `GPSAnchor`（不變）+ edge 平移殘差
    `(t_dst + R_dst @ t_rel) − t_src`——這時候 `R_dst` 是已知常數（來自
    `GimbalYawDegree`），不再是決策變數，**耦合問題從根本上消失**
    （機制 (2) 的耦合正是因為 `R_dst` 是決策變數，一旦變成常數，
    `R_dst @ t_rel` 這一項對平移殘差來說純粹是個常數平移量，整個平移
    子問題退化成一個乾淨的線性最小平方法）。`gimbal_yaw` 沒提供時，
    才退回現有的聯合求解（明確標記為降級路徑，品質沒有保證，呼應上面
    已經記錄的「`YawAnchor` 依賴風險」）。這比兩階段最小平方法簡單
    得多——不是「先解一次最佳化、再解第二次」，是「旋轉直接用感測器
    讀數、完全不最佳化，只對平移做最佳化」。**這個方向目前只是構想，
    還沒有評估範圍/可行性，是下一步要做的事**。
- **`run_pipeline` 端到端跑完真實 `data/smoke/` 10 張影像後，發現目前整套
  `pipeline_status`/`successful_image_count` 分類規則，加上 task2.md
  定義的 `distortion` metric，對「pose graph 收斂後產生的 scale 崩潰」
  這類幾何異常完全沒有偵測能力——這比「按設計運作」更值得明確記錄，
  不只是順便一提**：
  - **分類規則的盲點**：方案 B 的 `_MIN_INLIERS_FOR_DETERMINED_
    HOMOGRAPHY=4` 門檻只判斷「有沒有足夠資料算出 homography」，
    不判斷「pose graph 最佳化後的幾何合不合理」——這是常數旁邊的
    docstring 本來就講清楚的界線，不是遺漏。真實資料上，node1/2/3
    的 `inlier_count`（40、40、717）都遠高於門檻 4，所以這次
    `run_pipeline` 端到端跑出 `pipeline_status=success`、全部 10 張
    都算成功，**即使 mosaic 裡確實包含已知的嚴重幾何扭曲**（node1
    scale=0.169、node2 scale=0.244，見上面 Check A/B/C 那一輪診斷）。
  - **`distortion` metric 的盲點**：`compute_distortion` 量的是局部
    Jacobian 的各向異性（`abs(log(sigma1/sigma2))`）。我們的
    `GlobalTransforms` 永遠是 Sim(2) 相似變換（均勻縮放+旋轉，沒有
    shear），這種變換的 Jacobian **恆有 `sigma1==sigma2`**，不管
    縮放係數本身是 0.169 還是 5.0——「各方向縮放一致」跟「縮放係數
    本身合不合理」是兩個獨立的性質，`distortion` 只檢查前者。這是
    這個 metric 定義本身的性質，不是 bug，但**必須記錄清楚原因，
    避免之後有人誤以為 `distortion≈0` 就代表幾何沒問題**——這次真實
    資料跑出 `distortion≈5e-15`（幾乎精確 0），但 node1/2/3 的 scale
    崩潰完全沒有被這個數字反映出來。
  - **結論：`metrics_df` 目前沒有任何一個欄位能反映「這個 node 的
    scale 崩潰了」這件事**——`reprojection_error_px`/`inlier_ratio`/
    `inlier_count` 只看 feature matching 品質，`distortion` 只看
    局部各向異性，`pipeline_status`/`successful_image_count` 只看
    「有沒有數據可以算」。三層規則各自的設計都合理，但疊起來就是
    對這一類問題視而不見。
- **新增獨立待辦（見下面「目前狀態」）：需要一個新的品質指標或分類
  規則層級，檢查 `GlobalTransforms.transforms` 裡每個 node 的
  `sqrt(a²+b²)`（scale）是否落在合理範圍**，而不是只看
  `inlier_count` 夠不夠或 Jacobian 各向異性。這個待辦排在「10 月
  重新評估旋轉退化問題」旁邊，兩者密切相關——等旋轉/scale 問題真正
  修好之後，這個檢測層級可以順便當驗證修復是否生效的工具（就像
  `blend.py` 那次的霧化色塊一樣，是一個額外的診斷手段，不只是防禦
  機制）。
- **canvas 超線性成長是旋轉/scale 退化的第二種代價，不是獨立問題——第一種
  代價是幾何扭曲本身（見上面），這次用真實記憶體診斷量化出第二種代價：
  記憶體**。用真實 `data/`（0299~0330 系列，52 張裡取樣）跑過一次純診斷
  （沒有改動 `src/sea_mosaic/` 任何程式碼）：沿同一條真實序列，把影像數量
  N 從 2 逐步加到 50，每個 N 都重新跑一次 `compose_global_transforms` +
  `compute_canvas_size`（刻意不呼叫 `warp_images`/`blend_images`——純幾何
  計算，不配置任何 canvas 尺寸的陣列，任何 N 都安全，可以放心跑到 50）。
  結果：canvas 面積從 N=2 的 55.1 Mpx 一路長到 N=50 的 293.0 Mpx（5.3
  倍），而且不是平滑成長——N=11~20 之間一度停滯在 171.2 Mpx（第一次看
  到這個現象時誤判是「已經收斂的平原」），但 N=27 之後又恢復劇烈成長，
  到 N=50 已經逼近 300 Mpx。這個忽停滯忽暴衝、非單調的模式，跟已經
  記錄的旋轉/scale 退化（節點的 scale 忽然崩潰或暴衝、旋轉角度不連續）
  是同一個根因在不同層面的表現：`compute_canvas_size` 算的是「包住每張
  影像四個角點投影後的最小外接矩形」，退化把某些節點的角點投影到離群的
  極端座標，canvas 只是如實反映這個離群程度，兩者不是各自獨立的兩個
  問題。
- **上面的 canvas 超線性成長，疊加 `warp_images`/`blend_images` 目前
  「每張影像的完整 canvas 尺寸陣列同時全部留在記憶體裡」的架構（記憶體
  用量是 O(N × canvas_size)，不是 O(canvas_size)），已經用真實 cgroup
  記憶體量測驗證是真實資料集在張數變多時被 OOM killer 殺掉的根因**
  （見上面「環境」段落：這個 pod 的真實記憶體上限是 60GB，不是 `free -h`
  顯示的主機總量）。實測 N=20（真的跑完 `warp_images` + `blend_images`，
  沒有被殺）cgroup 峰值用量 ≈54GB，跟用「`N×canvas×4bytes`（`blend_images`
  裡的 `weights` 字典跟 `seam_masks` 字典，在迴圈跑完的當下兩者是同時
  存活的，不是先釋放一個再建另一個）+ `canvas×8bytes`（`total_weight`）
  + `canvas×24bytes`（`mosaic` 累加器）+ 單次迭代的暫態 float64 陣列」
  推導出的公式預測值（54.8GB）幾乎精確吻合，驗證了這個記憶體模型是對的。
  用同一個模型外插（純公式推算，沒有真的跑，因為預測值遠超這個 pod 的
  上限，真的跑會有被 OOM killer 殺掉、影響這個共用 pod 上其他人的真實
  風險）：這條真實序列的記憶體用量大約在 N≈22~23 就會超過 60GB 上限，
  N=50 時推算高達 ≈199GB——「50 張被 Killed」不是因為 50 這個數字本身
  特別，是因為這批資料的 canvas 早在 N≈22~23 附近就已經逼近上限，50
  只是遠遠超過那個早就存在的臨界點。這組具體數字（N≈22~23、199GB）全部
  綁定這批舊資料自己的 canvas 成長曲線，10 月正式資料集的張數（幾百幾千
  張規模）、飛行模式、退化程度都可能完全不同，不能直接沿用這裡的具體
  數字去預測正式資料集會在第幾張撐不住。但**兩個結構性結論會沿用，不
  受資料集換掉影響**：(1) O(N × canvas_size) 的記憶體架構，只要資料量
  夠大，必然會撞上某個記憶體上限，不管那個上限是 60GB 還是別的數字——
  這是架構本身的問題，不是這批資料特有的；(2) 如果旋轉/scale 退化問題
  到 10 月依然存在，canvas 本身的失控成長會讓這個上限被提前撞上——把
  記憶體架構改成 O(canvas_size)（不再隨 N 成長，見下面「目前狀態」的
  streaming accumulator 待辦，已完成實作並接進 `run_pipeline`）解決的
  是問題 (1)，不是問題 (2)；問題 (2) 仍然要靠旋轉/scale 退化本身被修好
  才能真正解決（10 月重新評估），兩者要分開處理，修好其中一個不代表
  另一個就不用管了
- **`io_utils.py` 的 `load_image`/`load_images` 仍是 `...` 空殼**（沒有真的用
  PIL/cv2 讀圖、也沒有測試覆蓋），已經在兩個不同任務裡各撞到一次：
  第一次是 SIFT `match_pair` 對照實驗（0352 vs 0353 inlier_ratio 診斷），
  第二次是 `compose_global_transforms` 全流程診斷（`data/smoke/` 10 張真實
  影像）。兩次都是繞過（在診斷腳本裡直接用 `cv2.imread` 讀圖，不動
  `io_utils.py`），因為色彩空間、失敗處理等設計決定不該在一次性診斷腳本
  裡定案。這代表這不是單次巧合，是一個會反覆卡住多個任務的真實缺口——
  `warp.py`/`blend.py` 大機率也需要讀取真實影像像素，很可能會第三次撞到。
  已在下面「目前狀態」把 `load_image`/`load_images` 明確排入待辦，排在
  `warp.py` 之前。

### GPS anchor 框架鏡射 bug 與分階段架構決定（2026-09-26）

**措辭原則：這一節用來解釋上面那條旋轉/scale 退化診斷鏈，不是推翻它。**
上面記錄的觀測數字（node1/2/3 的 scale 0.169/0.244/0.226、逐邊隔離時各邊都偏
70°～97°、decoupled 公式沒有改善、YawAnchor 在 weight 0.02～10.0 都拉不動、
純旋轉鏈 Check A 的 scale 崩潰）都是真實量到的，這次也重現了其中的 GPS-anchored
崩潰，數字到小數點後三位都吻合。改變的是**對這些觀測的歸因**：大部分現象都有
一個先前沒被發現的共同原因，就是 GPS anchor 的座標框架跟 mosaic 座標框架差了
一個鏡射。「這批替代資料的 homography 精度不足以支撐幾何最佳化」這個歸因，
現在已經不是主要解釋（見下面驗證 1：homography 位移跟正確框架下的 GPS 預測
中位數只差約 16 px）。

- **bug 本身（`posegraph.build_pose_graph`，GPS anchor 建構那段）**：把
  `ppm × (East, North)` 直接當成 mosaic 的 `(x, y)`，有三個錯：
  1. **y 軸方向錯**：影像 y 軸朝下，North 必須取負號。
  2. **少轉參考影像的偏航**：mosaic 座標系就是參考影像自己的座標系（reference
     pose = I），GPS 位移要先依 `GimbalYawDegree(ref)` 旋轉，才能跟 mosaic 座標比較。
  3. **anchor 綁錯點**：anchor 綁在 `pose[:2,2]`（影像左上角被映射到的位置），
     但 GPS 量的是影像中心。影像轉 175° 時，這個落差約 5000 px（≈177 m）。
  正確的換算（世界位移 `d=(dE,dN)` 公尺，參考影像 yaw ψ，順時針自北起算）：
  `r=(cosψ, −sinψ)`、`f=(sinψ, cosψ)`，`x = ppm·(d·r)`、`y = −ppm·(d·f)`，
  而且對應的是影像中心 `pose @ (W/2, H/2)`，不是 `pose[:2,2]`。
  從程式碼現用的 anchor 換到正確 anchor，最佳線性映射的 **det = −1.0**（實測，
  參考影像 yaw=70.8°）——是鏡射，不是任何旋轉。Sim(2) 的 `(a,b)` 只能表示
  旋轉＋均勻縮放，表示不了鏡射；最佳化器只能用「把 scale 壓扁、角度亂轉」去
  妥協一個幾何上不可能滿足的目標。
- **驗證 1（慣例檢驗）**：`data/` 0350～0361（index 39～50，跨越 0352～0354 的
  U 型迴轉，也就是上面 node1/2/3 崩潰的那一段）11 條相鄰邊。這次用 SIFT +
  **FLANN** knnMatch + ratio 0.75（為了控制在 1 分鐘內），不是先前的 BFMatcher，
  所以 inlier_count 跟上面表格有小幅差異（例如 42/48/42/705，對照 BF 的
  40/40/717）。用 `inv(H) @ center − center` 量 dst 影像中心在 src 影像座標系裡
  的位移，跟 GPS 預測比較：

  | 假設 | 11 條邊的誤差（px） |
  |---|---|
  | y = −N，並依 src 的 gimbal yaw 旋轉（正確慣例） | 2～107（中位數約 16） |
  | y = +N（沒有南北翻轉） | 493～799 |

- **驗證 2（修正 anchor 後的最佳化）**：同一個 12 張子集，用獨立的診斷 solver
  （參數化成 θ, s, tx, ty；edge 殘差與 information 權重跟正式程式碼相同；
  無邊界時的輸出已確認跟正式的 `optimize_pose_graph` 完全一致）：

  | 設定 | scale 範圍 | 最大角度誤差（對照 GimbalYawDegree 差值） |
  |---|---|---|
  | 目前程式碼（重現崩潰） | 0.17～1.08 | 156° |
  | 目前 anchor ＋ scale 邊界 [0.8,1.2] | 0.80～1.03 | 156°（scale 卡在邊界，角度完全沒改善） |
  | 目前 anchor ＋ gimbal yaw 初始化 | 0.17～1.08 | 156°（崩潰解是穩定最小值，跟初始值無關） |
  | 修正 anchor，無邊界 | 0.65～1.00 | 18°（迴轉段 node 誤差 0.3～6°） |
  | 修正 anchor ＋ 邊界 [0.8,1.2]，θ=0 初始化 | 0.80～1.16 | 118°（卡在壞的局部解） |
  | 修正 anchor ＋ 邊界 [0.8,1.2] ＋ gimbal yaw 初始化 | 0.80～1.03 | 7.8° |

- **哪些舊觀測被這個 bug 解釋、哪些沒有**：
  - 已解釋（有重現）：GPS-anchored 的 node1/2/3 崩潰——修正 anchor 後，迴轉段
    角度誤差降到 0.3～6°。
  - **已驗證（補跑，2026-09-26）：逐邊隔離時每條邊都獨立退化**。同一個逐邊隔離
    實驗（只留 src、dst 兩個 node ＋ 各自 GPS anchor ＋ 這一條邊，
    `reference_index=src`），對 0350～0361 的 11 條邊各跑兩次。「目前 anchor」
    走正式的 `build_pose_graph` ＋ `optimize_pose_graph`；「修正 anchor」走診斷
    solver，只換 anchor 框架，edge、information、初始值（θ=0、s=1）都相同：

    | edge | inlier | 真實相對偏航 | 目前 anchor：角度 / scale / 誤差 | 修正 anchor：角度 / scale / 誤差 |
    |---|---:|---:|---|---|
    | 0350→0351 | 23 | 0.0° | 79.8° / 0.988 / 79.8° | 0.3° / 0.987 / 0.3° |
    | 0351→0352 | 42 | −31.8° | −100.2° / 0.203 / 68.4° | −30.1° / 0.919 / 1.7° |
    | 0352→0353 | 48 | −62.2° | −59.6° / 0.171 / 2.6° | −62.5° / 0.953 / 0.3° |
    | 0353→0354 | 42 | −31.4° | −13.9° / 0.244 / 17.5° | −30.2° / 0.980 / 1.2° |
    | 0354→0355 | 705 | −49.7° | 28.8° / 0.226 / 78.5° | −46.6° / 0.946 / 3.1° |
    | 0355→0356 | 1597 | 0.0° | −97.1° / 1.076 / 97.1° | −0.2° / 0.975 / 0.2° |
    | 0356→0357 | 1979 | 0.0° | −88.6° / 0.989 / 88.6° | −0.0° / 1.005 / 0.0° |
    | 0357→0358 | 2329 | 0.0° | −89.0° / 1.026 / 89.0° | −0.3° / 1.001 / 0.3° |
    | 0358→0359 | 2144 | 0.0° | −75.7° / 0.842 / 75.7° | 0.3° / 1.058 / 0.3° |
    | 0359→0360 | 2411 | 0.0° | −70.7° / 0.836 / 70.7° | −0.3° / 1.077 / 0.3° |
    | 0360→0361 | 2768 | 0.0° | −73.3° / 0.772 / 73.3° | 0.7° / 1.069 / 0.7° |

    「目前 anchor」重現了舊的隔離表格（例如 0352→0353 的 −59.6°/0.171，對照舊的
    −59.435°/0.1690；差異來自 FLANN 與 BF 的配對不同）。修正 anchor 後，11 條邊的
    誤差全部 ≤3.1°、scale 落在 0.92～1.08。**逐邊退化確實是鏡射 bug 造成的**，而
    不是 homography 本身的問題。附帶修正先前的描述：直線段的偏移是 70.7°～97.1°，
    不是整齊的「約 90°」。
  - **已驗證（補跑，2026-09-26），但結果跟預測的形式不同：YawAnchor 拉不動**。
    用 0352～0361（ref=0352，node k 就是舊 smoke 的 node k，跟舊的 weight sweep
    同一組 9 條邊），`build_pose_graph` 產生的 YawAnchor 殘差不變，weight 掃
    None/0.02/0.5/1/2/5/10：
    - 目前 anchor：重現了舊的「拉不動」。node3 誤差在所有 weight 下都是 172.1°；
      node2 從 79.7° 到 weight=10 才降到 48.2°；node4～9 固定在 46°～73°；
      node1～3 的 scale 固定在 0.17～0.27。
    - 修正 anchor：**不加 YawAnchor 時，node1～9 的誤差就只有 0.3°～5.9°**
      （node2 0.7°、node3 3.7°），沒有東西需要被「拉回」。加上 YawAnchor 後，
      拉力是單調、看得到的：weight=10 時最大誤差降到 2.9°（node9 從 5.9° 降到
      1.6°），scale 從 0.80～0.98 提升到 0.88～1.06。但 `weight=0.02` 仍然幾乎
      沒有作用（跟不加完全相同），要到 weight≈5 才明顯。
    - **結論**：預測的形式是「修正後 YawAnchor 會把 node2/3 拉回來」，實際上是
      「修正後 node2/3 本來就是對的，不需要 YawAnchor 救」。「之前拉不動是因為在跟
      不可能的目標拔河」這個推論仍然得到支持：拿掉不可能的目標後，同一個
      YawAnchor 的拉力就能正常、單調地反映在結果上；在錯的框架下，weight=10 都
      推不動 node3。另外，修正 anchor 後仍殘留 scale 漂移（0.80～0.98，Check A
      的機制），YawAnchor 在 weight≥5 時可以部分抵消。
  - 仍是推論，沒有單獨重跑：decoupled 公式沒有改善（問題不在耦合寫法，而在
    anchor 目標本身不可能滿足）。上面兩項驗證都是用 decoupled 公式跑的，所以
    「decoupled 公式在正確 anchor 下運作正常」這一點已經成立；但舊的「耦合」公式
    在正確 anchor 下會不會一樣好，沒有測過。這不影響現在的決定，因為舊公式已經
    不在 codebase 裡。
  - **沒有被解釋、仍然成立**：Check A 的純旋轉鏈 scale 崩潰（0.13～0.57）。那次
    完全沒有 GPS，原因是 `(a,b)` 參數化本身帶著 scale 自由度，多條邊各自輕微
    不完美的 2×2 子矩陣在聯合求解時把 scale 複合壓扁。修正 anchor 後殘留的
    scale 漂移（0.65～0.99）是同一個機制。這是分階段架構的 Stage A 刻意用
    「參數化裡根本沒有 scale」的方法處理的原因。
  - 未重新量測：canvas 超線性成長（很可能是退化的下游後果，修正後要重量）。
- **資料模式（影響之後的配對策略）**：`data/` 的 52 張是蛇形測繪，不是單一
  直線：3 條來回航線（0299～0304、0317～0351、0355～0362），航向在
  GimbalYaw −104.3° 與 +70.8° 之間切換（相差 175°，朝向分布是雙峰的），
  0352～0354 是 U 型迴轉。航線間距約 27 m，地面覆蓋寬約 141 m，旁向重疊約 80%，
  所以**跨航線的 loop closure 真實存在**，只用 `sequential_pairs()` 會完全浪費掉。
  另外，`data/smoke/` 目前是 9 張（0299～0304 + 0317～0319，中間跨 90 m 缺口與
  175° 轉向），已經不是上面各條記錄提到的那組「10 張連續飛行序列」。
  700 張的新資料集目前不在這台機器上，飛行模式尚未確認。
- **決定：用分階段架構取代一次性聯合最佳化**（每個 stage 都要有自己獨立的
  測試，不能等 Stage D 跑完才發現前面哪裡錯）：
  - **Stage A：純旋轉平均**——只吃 edge 的相對旋轉，不碰 GPS、不碰 GimbalYawDegree。
  - **Stage B：GPS 直接擺放**——修正上面三個 anchor 框架錯誤，直接算出每個 node
    的位置（不是疊代求解），縮放依飛行高度（`pixels_per_meter`）。
  - **Stage C：座標系對齊**——Stage A 的旋轉基準是任意的，要找一個全域旋轉偏移，
    把它對齊到 Stage B 的座標系。
  - **Stage D：有界小幅精修**——從前三步的乾淨初始解出發，把 edge 殘差、
    弱權重 GPSAnchor、YawAnchor 放進同一個目標函數，修正量要有界（例如 scale
    ±10～20%）。
- **Stage A 的設計決定（已定案）**：
  1. **譜方法**（每個 node 一個單位複數，組 Hermitian 矩陣取主特徵向量，再把
     每個分量正規化成單位長度），不用 chordal `least_squares`。理由：不需要初始
     值（閉式解）、參數化裡沒有 scale 自由度、能真正利用蛇形資料的跨航線 loop。
     **連通分量檢查是必要防護**：先切出連通分量、各自求解並標記出來；圖不連通時
     各分量的相位互相無關，不能靜默產出垃圾結果。
  2. **yaw 完全不進 Stage A**，函式簽名裡不出現 yaw。GimbalYawDegree 只在
     Stage C（對齊）與 Stage D（YawAnchor）出現。這也保留了一條不依賴 DJI 專屬
     XMP 欄位的旋轉路徑（呼應「YawAnchor 依賴風險」待辦）。
  3. **新模組 `src/sea_mosaic/rotation_averaging.py`**，不塞進 `posegraph.py`，
     輸入只需要 `(src, dst, θ_ab, weight)`。
  4. **既有 `optimize_pose_graph` 暫時不改名、不移除**。Stage A 是純新增，真正
     切換發生在 Stage D 完成、`compose_global_transforms` 重新接線的時候。既有
     測試在 Stage A 期間全部原樣保留。之後會受影響的：
     `test_build_pose_graph_creates_gps_anchors_in_pixel_units`、
     `test_build_pose_graph_sets_initial_pose_translation_from_gps_anchor`
     （目前鎖定的正是錯的 `(E,N)` 慣例，Stage B 時要改）；YawAnchor weight 的
     能力邊界測試（在舊 anchor 框架下校準，Stage D 時重新評估）；
     `tests/fixtures/golden_pipeline/`（重新接線時會變）。
  - **待注意**：只有序列相鄰邊（圖是一棵樹）時，任何旋轉平均的解在數學上都等於
    沿鏈累加角度。這不違反硬性架構約束 4（約束 4 針對的是完整 homography 連乘，
    平移仍由 GPS 決定），但要記得 Stage A 在沒有 loop 的資料上本質就是角度累加。
    θ_ab 的取法（沿用 `relative_pose`，或在影像中心取 Jacobian 做 polar
    decomposition）與 edge→node 的角度正負號，要用獨立推導的期望值測試，不能用
    同一個函式自我比對（`_yaw_target_vector` 符號 bug 的教訓）。
- **matcher 不在這次一起換**：LoRetta 等其他 matcher 留到 Stage A～D 全部完成、
  架構穩定之後，當作獨立的 matcher 比較實驗。架構修正還沒完成時不要同時換兩個
  變因，否則出問題時分不清是架構還是 matcher 的問題。
- **因此失效或待重新驗證的舊結論**：上面「已知的暫緩事項」的「旋轉退化暫緩到
  10 月」（改為現在就用分階段架構處理）；YawAnchor `weight≈0.02` 與相關能力邊界
  （在錯的 anchor 框架下校準）；「`GimbalYawDegree` 是唯一穩定的旋轉訊號來源」
  （在錯的 anchor 框架下得出；修正後 edge 自己的旋轉在迴轉段就準到 0.3～6°）。
  `pixels_per_meter≈28.703` 不受影響（純幾何推導）。`inlier_count_reference`
  的 median 取法本身不變，但它「讓 information 發揮設計意圖」的結論要在新架構下
  重新驗證。

### streaming accumulator 記憶體重構回顧（總結）

上面關於 canvas 超線性成長、cgroup 記憶體上限、streaming accumulator 設計與
實作的記錄橫跨好幾條很長的條目，這裡整理成一個精簡的時間線摘要，方便之後
快速回顧整件事的來龍去脈，不用重新讀完上面所有細節。

1. **觸發原因**：50 張真實資料壓力測試被 OOM killer 殺掉。診斷發現這個 pod
   的真實記憶體上限是 cgroup `memory.max=60GB`，不是 `free -h` 顯示的主機
   總量（314GB）——`free -h` 看到的是整台共用主機的記憶體，跟這個 pod 實際
   能用的量無關，會嚴重誤導記憶體判斷。**之後任何記憶體相關的診斷或容量
   規劃都要看 `/sys/fs/cgroup/memory.current` 對照 `/sys/fs/cgroup/
   memory.max`，不要再看 `free -h`**（見上面「環境」段落）。

2. **根因**：`warp_images`/`blend_images` 把每張影像的全畫布尺寸陣列
   （warped image、warped mask、distance-weight、seam mask）同時留在記憶體
   裡，複雜度是 O(N × canvas_size)，不是 O(canvas_size)。而 canvas_size
   本身又會因為已經記錄過的 pose-graph 旋轉/scale 退化問題超線性成長（見
   上面「已知的限制」的 Effect A/B 診斷：canvas 面積從 N=2 的 55.1 Mpx 長到
   N=50 的 293.0 Mpx，非單調、忽停滯忽暴衝）。**這是第一次量到這個既有
   已知 bug 的記憶體代價，退化問題本身不是這次新發現的**——旋轉/scale
   退化早就記錄在案，只是先前只知道它會造成幾何扭曲，沒意識到它還會拖垮
   記憶體。

3. **修法**：streaming accumulator 設計，分四階段實作，每階段都先寫等價性
   （`np.array_equal`，不是近似相等）/laziness（call-counting proxy 證明
   真正逐張處理，不是建好 list 再假裝 streaming）/gc（weakref 證明已處理過
   的陣列沒有被暗中續留）/`tracemalloc` 平坦度（N=10 vs N=100 實測峰值比值
   接近 1，不是接近 10）測試確認紅燈，再實作到綠燈，全程不允許既有測試
   迴歸：
   - `warp_images_streaming`（generator，逐張 yield，不一次 materialize
     全部 N 張）
   - `blend_images_streaming`（`weighted_sum`/`weight_sum` 兩個固定大小
     累加陣列，單一 pass 折入即可丟棄，只回傳 mosaic 不回傳 `seam_masks`）
   - `compute_seam_error_streaming`（bounding-box 預篩選候選對，把成本從
     O(N²) 全對比較降到跟真正重疊的候選對數量相關，並正確處理 loop
     closure——不假設只有序列相鄰的影像會重疊；對候選對隨需重新 warp，
     一次只有 2 張影像的 canvas 尺寸資料存在）
   - 三段最後整合進 `run_pipeline`：原本「warp 全部 N 張 → 分類 → 只
     blend 通過的」三個分開的、都需要 N 張 canvas 尺寸資料同時存在的階段，
     融合成一個 streaming pass，失敗影像的 warped 陣列在分類的當下就被
     捨棄，不會進入 blend 的累加器；用 golden-fixture regression（3 個
     既有合成情境，重構前後比對 mosaic + metrics_df）驗證整條 pipeline
     輸出沒有改變。

4. **過程中的技術細節**：`blend_images`（eager）在正規化 alpha 時會多一次
   `.astype(float32)` 降精度，`blend_images_streaming` 全程 float64、最後
   才除一次——兩者數學上等價但不保證逐位元組相等。這造成了一個**自然發生
   （不是刻意構造）**的 1-ULP 舍入邊界差異：golden regression 測試裡單一
   像素 `(row=50, col=27)`，3 個 channel 皆從 `[2,2,2]` 變成 `[3,3,3]`，
   真值 ≈2.50000015，剛好落在 `.5` 邊界兩側。因為 streaming 是往後
   `run_pipeline` 實際會用的路徑，這個 golden 基準已經重新捕捉以反映
   streaming 版本的正確行為（另外兩個沒有踩到這個邊界的 golden 檔案維持
   原樣，byte-for-byte 驗證過未被觸碰）。

5. **新發現的獨立待辦**：驗證這次重構的端到端記憶體測試意外發現
   `compose_global_transforms` 的 `scipy.optimize.least_squares` 求解器
   本身有一個完全獨立、相當可觀的記憶體成本——N=100（合成資料，極小
   canvas）時單獨佔 15,948KB，是同一次 `run_pipeline` 呼叫總 peak
   （16,388KB）的 97%。懷疑根因是 Jacobian 矩陣用稠密（dense）方式建構，
   大小隨 pose-graph 參數量（隨 N 成長）平方增長，但這只是懷疑，還沒有
   像這次 streaming 重構一樣做過根因驗證。可能的修法方向是改用稀疏
   （sparse）Jacobian——pose graph 天生稀疏，多數 node 只跟少數相鄰 node
   有邊相連，一個 node 的殘差不會依賴大多數其他 node 的參數，理論上很適合
   稀疏表示法，但這個方向也還沒有實際評估可行性。已排入待辦（見下面
   「目前狀態」），但不是現在處理。

6. **結論**：架構複雜度從 O(N × canvas_size) 降到 O(canvas_size)，理論上
   不再受影像張數本身限制。**但這個結論只涵蓋 warp/blend/seam_error 三段，
   不涵蓋 `compose_global_transforms`**——那一段的複雜度還沒有被驗證過，
   10 月真正上大規模正式資料集之前，仍然需要對它做獨立的評估（可能還需要
   一次跟這次規模相近的診斷+設計+TDD 實作流程）。

## 目前狀態
- [x] SSH + VS Code Remote-SSH + Claude Code CLI 環境
- [x] 專案骨架
- [x] metrics.py + unit tests
- [x] EXIF/XMP 解析 (GPS 座標讀取 + 局部平面投影，21/21 tests passing)
- [x] io_utils.py: load_gimbal_yaw（XMP-only，無 EXIF 對應項）+
  posegraph.py: YawAnchor（設計、weight 量級驗證、TDD 實作皆完成，
  86/86 tests passing，見下面 feature-based pipeline 清單與上面
  「已知的限制」）
- [ ] direct georeferencing (geo/camera.py, geo/direct.py) 仍暫緩，見「已知的暫緩事項」
- [ ] feature-based pipeline
  - [x] estimate.py: sequential_pairs + match_pair/estimate_all_pairs
    (SIFT + RANSAC, 43/43 tests passing)
  - [x] posegraph.py: optimize_pose_graph (Sim(2) 最小二乘 + GPS anchor，
    45/45 tests passing，另用不對稱合成資料驗證過 node index 對應正確)
  - [x] posegraph.py: build_pose_graph (從真實 PairResult + GPS 座標建圖)
  - [x] compose.py: compose_global_transforms
  - [x] io_utils.py: load_gimbal_yaw（XMP-only，63/63 tests passing）
  - [x] posegraph.py: YawAnchor TDD 實作 —— `geo/projection.py` 新增
    `project_gimbal_yaw_degrees`；`posegraph.py` 新增 `YawAnchor`
    dataclass、`PoseGraphNode.yaw_anchor`/`PoseGraph.yaw_anchors` 欄位、
    私有 helper `_yaw_target_vector`（符號翻轉 + 單位向量）；
    `build_pose_graph` 新增 `gimbal_yaw`/`yaw_anchor_weight` 參數（含
    執行期必填檢查）；`optimize_pose_graph` 的 `residuals()` 接上
    yaw anchor 殘差項。88/88 tests passing（新增 24 個：
    `test_projection.py` 4 個、`test_posegraph.py` 18 個、
    `test_compose.py` 2 個），含兩個明確鎖定的能力邊界測試——優雅退化
    （`gimbal_yaw=None`/局部缺失時不報錯、`yaw_anchors=[]` 時行為跟之前
    完全一致）與 `weight≈0.02` 救不回嚴重旋轉錯誤（見上面「已知的
    限制」）。**用真實 `data/smoke/` 資料驗證時發現一個實作疏漏**：
    `compose.py` 的 `compose_global_transforms`（公開介面，`pipeline.py`
    未來會呼叫的入口）當初沒有跟著更新去接收/轉發 `gimbal_yaw`/
    `yaw_anchor_weight` 給 `build_pose_graph`，導致 `YawAnchor` 雖然在
    `posegraph.py` 層級測試全綠，透過公開介面卻完全用不到（`TypeError`）
    ——這正是「只看合成測試綠燈不夠」的活教材，已補上轉發邏輯與對應
    2 個測試（`ValueError` 轉發、真實 rescue 效果透過完整路徑驗證）。
  - [x] posegraph.py: 解耦 optimize_pose_graph 的殘差公式（旋轉/平移
    拆開，`R_dst @ R_rel − R_src` / `(t_dst + R_dst @ t_rel) − t_src`，
    全程不對決策變數取逆）—— **已實作，88/88 測試全綠，但用真實資料
    驗證後確認沒有解決原始退化問題**（node1/2/3 跟修正前幾乎一模一樣），
    根因比預期更深：`R_dst` 出現在平移殘差裡很可能是幾何上必然的（要用
    node 的旋轉把 edge 的平移向量轉到共同座標系才能比較），問題不在
    公式怎麼寫，在「單一聯合最小二乘法同時求解旋轉+平移」這個框架本身
    有結構性限制。完整推導、真實資料驗證數據、`old_diff = -inv(T_dst) @
    new_diff` 的代數關係都記在上面「已知的限制」。**這個修正保留在
    codebase 裡**（公式更乾淨、無明確 `1/scale²` 病態項，沒有壞處），
    但不能當作 root cause 已解決
  - [x] ~~posegraph.py: 兩階段求解（Rotation Averaging 再 Translation）~~
    —— **已評估並放棄**：範圍評估完成後，動手前先驗證了「第一階段
    純旋轉本身會不會有自己的退化模式」這個疑點，結果發現真的有（純
    edge 鏈在無 `YawAnchor` 時 scale 崩潰到 0.13～0.57，`weight≈0.02`
    救不動，需要 `weight≈1.0` 才乾淨），而且沒有 `GimbalYawDegree` 時
    退化範圍比現有聯合公式更廣。加上前面已經驗證過的兩個機制（GPS
    anchor 透過 `inv()` 污染旋轉、decoupled 公式仍透過 `R_dst @ t_rel`
    耦合），三個獨立機制指向同一個結論：**這批資料的 homography 精度
    天生不足以支撐純幾何聯合最佳化，`GimbalYawDegree` 是唯一穩定的
    旋轉訊號來源**——不管求解框架怎麼設計（聯合、decoupled、兩階段），
    只要旋轉還是決策變數，都會有退化風險。完整診斷數據見上面「已知的
    限制」
  - [ ] **分階段架構（取代一次性聯合最佳化，2026-09-26 定案）**——見「已知的
    限制」的「GPS anchor 框架鏡射 bug 與分階段架構決定」。取代下面那條「旋轉
    退化（暫緩）」。每個 stage 各自 TDD、各自有獨立測試：
    - [ ] Stage A: `rotation_averaging.py` 譜方法純旋轉平均（含連通分量檢查）
    - [ ] Stage B: GPS 直接擺放（修正 anchor 框架：y=−N、依參考影像 yaw 旋轉、
      anchor 綁影像中心）
    - [ ] Stage C: Stage A 旋轉對齊到 Stage B 座標系（全域旋轉偏移）
    - [ ] Stage D: 有界小幅精修（edge + 弱 GPSAnchor + YawAnchor），接回
      `compose_global_transforms`
    - [ ] 之後：跨航線配對（蛇形資料的 loop closure）、LoRetta 等 matcher 比較實驗
  - [ ] ~~posegraph.py: `compose_global_transforms` 的旋轉退化（暫緩）~~ ——
    **已被上面的分階段架構取代（2026-09-26）**，原文保留作紀錄：
    **暫緩，等 10 月正式資料集重新評估，不是現在要修的 bug**（見上面
    「已知的暫緩事項」的完整理由：合成資料已證明架構本身邏輯正確，
    問題根源是這批網路替代資料的 homography 估計精度不足，不是寫錯）。
    兩個候選解法（兩階段求解——已評估後放棄；旋轉直接採用
    `GimbalYawDegree` 常數、只對平移做最佳化——已有構想但未評估）都
    先不動手，留到 10 月拿到正式資料、重新跑一次「純旋轉鏈複合退化」
    診斷（見上面「已知的限制」Check A）之後再決定。`warp.py`/`blend.py`
    開發時要記得目前 `compose_global_transforms` 的旋轉輸出不可靠，
    下游先只用平移座標做粗略排列驗證，不依賴精確旋轉結果
  - [ ] metrics.py/pipeline.py: 新增偵測「node scale 崩潰」的品質指標
    或分類規則層級——**跟上面「旋轉退化（暫緩）」密切相關,排在它
    旁邊,但這次不處理**。見上面「已知的限制」：目前 `pipeline_status`/
    `successful_image_count`（方案 B 的 inlier 門檻）跟 `distortion`
    metric（Sim(2) 相似變換的 Jacobian 各向異性恆為 1，量不到均勻縮放
    崩潰）加起來，對「node1 scale=0.169 這種嚴重幾何扭曲」完全沒有
    偵測能力——真實資料端到端跑出 `pipeline_status=success`、
    `distortion≈5e-15`，但 mosaic 裡確實有已知的嚴重扭曲,`metrics_df`
    没有任何欄位反映這件事。具體構想：檢查
    `GlobalTransforms.transforms` 裡每個 node 的 `sqrt(a²+b²)` 是否
    落在合理範圍（例如遠離 1.0 就標記）。等旋轉/scale 問題修好後，
    這個檢測層級可以順便當驗證修復是否生效的診斷工具，不只是防禦
    機制
  - [ ] posegraph.py: optimize_pose_graph 的 information 單位失衡 ——
    獨立於 YawAnchor 之外的架構問題（見上面「已知的限制」根因 2）：
    `information = coef * eye(6)` 對混合了平移（像素單位，量級
    500～2500）與旋轉/縮放（無因次，量級 0.02～1.5）的 6 維殘差套用
    同一個純量係數，導致不管係數設多少，edge 對旋轉分量的約束力永遠
    比對平移分量弱 3～4 個數量級。這次刻意不修（YawAnchor 是先解決
    眼前 bug 的獨立手段，不是這個問題的根本解），排在 YawAnchor 實作
    之後、`load_image`/`load_images` 之前——因為這是 pose-graph
    本身的正確性問題，比單純的 IO 空殼更貼近 warp.py 依賴的核心邏輯。
    跟上面的旋轉退化是同一條診斷鏈的一部分，一併暫緩到 10 月重新評估，
    不單獨處理
  - [ ] posegraph.py: YawAnchor 依賴風險 —— YawAnchor 落地後會是旋轉
    分量的主要、甚至唯一有效約束來源（上面 information 單位失衡問題
    的直接後果），旋轉分量的可靠性因此幾乎完全綁定 `GimbalYawDegree`
    這個 DJI 專屬 XMP 欄位存不存在。需要在 build_pose_graph 或更上層
    加一個明確的偵測/警告機制（例如 `gimbal_yaw=None` 時記一筆 log 或
    某種狀態旗標），而不是讓旋轉分量安靜地退回無約束狀態卻沒人知道。
    10 月正式資料集如果換了 metadata 格式，這個風險會直接浮現，必須在
    那之前有个機制能察覺。同樣暫緩到 10 月一併重新評估
  - [ ] io_utils.py: load_image/load_images 正式實作（含測試）—— 目前是
    `...` 空殼，已反覆在多個診斷任務裡被繞過（見上面「已知的限制」），
    排在 warp.py 之前，因為 warp.py 大機率也依賴它
  - [x] warp.py: compute_canvas_size / warp_images —— 12/12 新測試通過
    （`tests/test_warp.py`，全部手構造合成資料，不需要真的讀圖，沒有碰
    `load_image`/`load_images` 空殼），100/100 全專案測試綠燈。刻意分層
    驗證：只測「給定的 3x3 transform 有沒有被正確套用」（含一組乾淨的
    合成旋轉案例），不測「`compose_global_transforms` 在真實資料上的
    旋轉準不準」——後者是上面「已知的暫緩事項」記錄的已知限制，這輪
    不處理。`camera_intrinsics`/`camera_poses` 仍是保留參數，未使用
  - [x] blend.py: blend_images —— distance-transform feathering（
    `cv2.distanceTransform(mask, DIST_L2, DIST_MASK_PRECISE)` 算每張
    影像的原始權重，正規化成 `alpha_i`，回傳的 seam mask 是連續值
    float32 [0,1]，不是二值分割）。9/9 新測試通過（`tests/test_blend.py`，
    全部手構造合成資料），109/109 全專案測試綠燈。跟 warp.py 同樣的
    分層驗證原則：合成資料驗證 blending 邏輯本身（含用鏡像對稱幾何
    精確驗證 50/50、用推導出的 1/17 門檻驗證接縫連續性，不是猜的
    數字），不追求在真實資料（已知旋轉不可靠）上產出視覺完美結果。
    **後續在規劃 `pipeline.py` 的錯誤隔離機制時發現並修正了一個
    `blend_images` 自己的真實邏輯漏洞**（不是資料品質問題，跟 compose
    的旋轉退化不是同一類、不能歸咎給這批替代資料）：`cv2.distanceTransform`
    對「完全沒有黑邊」的 mask（非零區域剛好佔滿整個陣列）沒有零像素可以
    量距離，會回傳一個溢位式的哨兵值（實測 ≈1.8e19），比正常影像的權重
    大 16 個數量級，會讓那張影像的 alpha 壓倒性主導、蓋掉所有真實幾何
    重疊資訊。這個情境本來就可能發生（例如只剩一張影像存活時，畫布剛好
    等於它自己的尺寸），`pipeline.py` 規劃的「隔離退化影像」機制還會
    提高這個情境出現的機率——**修法**：把 `distanceTransform` 的輸出
    clip 到該 mask 自己的對角線長度（`np.hypot(*mask.shape)`）——真實
    （有黑邊）mask 算出來的距離值最多在自身較短邊一半左右，對角線是
    一個寬鬆但有物理意義的安全上界，不會誤裁到任何真實數值，只會壓制
    這個哨兵值。已補 1 條回歸測試
    （`test_blend_images_no_black_border_mask_does_not_overwhelm_other_images`，
    用實測算出的精確數字 0.8333/0.1667 鎖定，不是猜的），110/110 全專案
    測試綠燈。
  - [x] pipeline.py: 串接 estimate → compose → warp → blend → evaluate metrics。
    **`run_pipeline` 的職責邊界已定案：收已讀好的 `images: dict[int,
    np.ndarray]`，不涉及檔案 IO**（原本骨架簽名是 `image_paths:
    list[Path]`，隱含要呼叫 `load_image`/`load_images`；已改成
    `images`，理由是「檔案讀取的色彩空間/批次失敗策略」跟「pipeline
    分類規則」是完全不同層次的問題，混在一起做出錯時無法定位，跟
    fake Matcher 測 RANSAC、合成資料測 warp 幾何是同一個分層原則）。
    `PipelineConfig` 相應擴充（`altitude_m`/`dfov_deg` 必填無預設值，
    直接在 `PipelineConfig()` 建構時就報錯，不用等 `run_pipeline` 跑到
    一半才發現缺東西；`gps_positions`/`gimbal_yaw`/`yaw_anchor_weight`/
    `pairs`/`loops` 維持可選、預設 `None`，跟 `compose_global_transforms`/
    `build_pose_graph` 自己的可選性一致）——`run_pipeline` 簽名統一走
    `config: PipelineConfig`，不跟散裝關鍵字參數混用兩種風格。14/14 新
    測試通過（`tests/test_pipeline.py`，用假 Matcher/`mock.patch`
    隔離,不需要真的 SIFT），126/126 全專案測試綠燈。
    - **成功判斷規則（方案 B）落地**：`i` 算成功 ⟺ `i` 在
      `global_transforms.transforms` 裡、warp 後 mask 非全零、且
      （`i==reference_index` 或 `i` 有 GPS anchor 或 `i` 至少一條連到
      它的邊 `inlier_count>=4`）。`_MIN_INLIERS_FOR_DETERMINED_
      HOMOGRAPHY=4` 的理由明確記在常數旁：4 點是 homography 8 個自由度
      在數學上的最低可解點數，**不代表「≥4 就可信」**（Check A 已經
      證明 inlier=717 這種遠高於門檻的邊一樣可能被複合放大成崩潰結果，
      「解得出來」跟「可信」是兩個不同的主張）。
    - **estimate 階段的錯誤隔離**：`run_pipeline` 自己逐 pair 呼叫
      `match_pair`、用 try/except 隔離（不改 `estimate.py`/
      `estimate_all_pairs` 本身的合約），`run_pipeline` 是系統邊界，
      底層函式維持單純。
    - **warp 階段的隔離改成事前過濾，不是 try/except**：實測
      `cv2.warpPerspective` 對任何病態矩陣（scale=0、NaN、inf）都
      **不拋例外**，只會靜默輸出全黑影像；但
      `compute_canvas_size` 對 NaN/inf 會拋 `ValueError`，而且如果
      canvas_size 是拿全部影像的 transform 一次算的，一張影像的
      NaN 會**連累其他健康影像的計算**。修法：在算 canvas_size 之前
      先用 `np.all(np.isfinite(transform))` 過濾掉非有限的
      transform，有限但退化（scale=0）的則沿用「warp 後 mask 非全零
      才算成功」的既有規則自然接住，不需要新機制。
    - **`compose_global_transforms` 的 `optimization_status !=
      "converged"`（含 `"not_converged"` 跟 `"failed"`）視同全部
      影像失敗**，完全不呼叫 `warp_images`/`blend_images`——不信任一個
      最佳化器自己都不認為收斂的結果。
    - **`pixels_per_meter`/`inlier_count_reference` 完全由
      `run_pipeline` 內部算出**（前者用 reference image 自己的
      `shape` + `config.altitude_m`/`dfov_deg`；後者用這次真正跑出來
      的 `pair_results`），不是呼叫端傳入的參數。
  - **實作過程中發現並修正了一個 `posegraph.py` 自己的真實 bug（不是
    `run_pipeline` 寫錯，是既有、已測試模組裡先前沒被抓到的邊界情況）**：
    `optimize_pose_graph` 的 `initial_params = np.concatenate([...])`
    是無條件執行的，執行順序在「圖裡沒有任何可最佳化 node（只有
    reference 自己）」這個分支的判斷之前——當這個情況真的發生時（單張
    影像輸入、或所有邊都失敗只剩 reference），`np.concatenate([])`
    直接拋 `ValueError`，那個本來就是為了處理這個情況而寫的分支永遠
    到不了。既有 27 條 `posegraph.py` 測試從來沒有測過「圖裡只有一個
    node」這個案例，所以沒被抓到。**修法**：把這行包進
    `if optimizable_indices: ... else: initial_params = np.zeros(0)`。
    已補 2 條回歸測試進 `tests/test_posegraph.py`（單一 reference
    node、無/有 anchor 兩種情況），29/29 該檔案測試綠燈，跟
    `blend_images` 那次一樣的原則：修正跟回歸測試進它自己的測試套件，
    不是靠上層 `run_pipeline` 繞過去。
  - [ ] pipeline.py: `run_pipeline` 要不要自動呼叫 `metrics.py` 已有的
    `save_metrics_txt`（存 `metrics.txt` 到跟 mosaic 圖片同目錄，per
    docs/task2.md §9）——**這輪刻意不處理，留給下一輪獨立討論**：
    要不要輸出、輸出到哪個路徑、跟 mosaic 圖片存在同一目錄的規則
    怎麼定、圖片本身要不要也是 `run_pipeline` 自動寫檔（目前
    `run_pipeline` 只回傳 `(mosaic, metrics_df)`，沒有寫任何檔案）,
    這些是新的、獨立的設計決定,不屬於「串接四段管線」這輪的範圍。
  - [x] warp.py/blend.py/metrics.py/pipeline.py: streaming accumulator 記憶體
    重構——把 `warp_images`/`blend_images`/`compute_seam_error` 的
    O(N × canvas_size) 記憶體用量降到 O(canvas_size)，不再隨影像數量 N
    成長，並接進 `run_pipeline` 取代舊的 eager 呼叫。分四階段、每階段
    先寫等價性/laziness/gc/tracemalloc 平坦度測試確認紅燈（因為新函式
    還不存在），再實作到綠燈，不允許既有測試迴歸：
    - `warp.py`: `warp_images_streaming`（generator，逐張 yield
      `(index, warped_image, warped_mask)`，canvas_size 由呼叫端用既有
      `compute_canvas_size` 算好傳入，不新增攜帶 canvas_size 的類別）。
      用 `cv2.warpPerspective` 的 call-counting proxy 證明真正逐張執行
      （不是先建 list 再 `yield from`），用 weakref+gc 證明已 yield 過的
      陣列沒有被任何隱藏快取續留。
    - `blend.py`: `blend_images_streaming`（`weighted_sum`/`weight_sum`
      兩個固定大小累加陣列，單一 pass 折入後即可丟棄每張影像的 canvas
      尺寸暫存陣列，數學上等價於現有「先正規化成 alpha 再加權平均」的
      兩階段做法，但只回傳 mosaic，不回傳 `seam_masks`——那是
      `seam_masks` 唯一的消費者 `compute_seam_error` 需要的東西，per
      docs/task2.md 本來就是可選欄位）。float64 除法結合律不保證逐位元組
      相等（除的順序不同），但用 `tracemalloc` 實測 N=10 vs N=100 peak
      比值 0.9993，記憶體確實打平。
    - `metrics.py`: 新增 `_image_canvas_bbox`/`_bboxes_overlap`（bounding
      box 預篩選候選對，edge-touching 採 inclusive 慣例，因為這個
      filter 唯一的正確性要求是「不能漏掉真的重疊」，多篩進來的候選對
      由既有 pixel-level overlap 檢查精確排除）+
      `_candidate_overlapping_pairs`（成本從跟 N² 相關降到跟真正重疊的
      候選對數量相關，同時正確處理 loop closure——不假設只有序列相鄰的
      影像會重疊，已用真實案例驗證非相鄰但確實重疊的 pair 有被正確納入）
      + `compute_seam_error_streaming`（對候選對隨需重新 warp，一次只有
      2 張影像的 canvas 尺寸資料存在，用真實情境驗證過：對子集合計算時
      不需要額外傳入完整集合的 canvas origin，因為 origin 只是均勻套用
      在所有比較影像上的純平移，不會改變任兩張影像的相對比對結果，只要
      canvas_size 夠大不會裁切即可——這點原本以為需要額外參數修正，
      後來用實測推翻了這個假設，沒有加不必要的參數）。
    - `pipeline.py`: `run_pipeline` 把原本「warp 全部 N 張 → 用完整
      `warped_masks` dict 做成功/失敗分類 → 只 blend 分類通過的影像」
      三個分開的、都需要 N 張 canvas 尺寸資料同時存在的階段，融合成一個
      streaming pass：`warp_images_streaming` 逐張 yield，一個包著它的
      filter generator（`_successful_only`）立刻對每張影像分類（
      是否為 reference/有沒有 determined edge/有沒有 GPS anchor 這些
      跟影像本身無關的判斷已經在迴圈外預先算好；只有「warped mask 是否
      非空」這一項真正需要當下的 warp 結果），失敗的影像的 warped
      陣列在那個當下就變成沒有任何引用、被捨棄，從來不會進入
      `blend_images_streaming` 的累加器，也沒有被存到別的地方；只有
      通過分類的才 yield 下去餵給 `blend_images_streaming`。
      `compute_seam_error` 的呼叫改成 `compute_seam_error_streaming`，
      結果 patch 進 `evaluate_stitching_metrics` 回傳的 `metrics_df`
      的 `seam_error` 欄位（`evaluate_stitching_metrics` 本身簽名完全
      不動，因為 `warped_images`/`warped_masks`/`seam_masks` 只有這
      一個用途，這是全部四階段裡唯一沒有改動任何既有公開函式簽名的
      設計）。用 golden-fixture regression（`tests/fixtures/
      golden_pipeline/`，3 個既有合成情境在重構前先各自存一份
      `mosaic.npy`+`metrics_df.pkl`，重構後比對）證明整條 pipeline
      輸出不變。**其中一個 golden 檔案（`isolated_node_without_gps_
      anchor`）重新捕捉過一次**：發現真實出現（不是刻意構造）的
      one-ULP 舍入邊界差異——單一像素 `(row=50, col=27)`，3 個 channel
      皆從 `[2,2,2]` 變成 `[3,3,3]`，真值 ≈2.50000015，兩種算法都對，
      只是 `blend_images`（eager）在正規化 alpha 時有一次
      `.astype(float32)` 降精度，`blend_images_streaming` 全程 float64、
      最後才除一次，剛好在這個像素落在 `.5` 邊界兩側——因為 streaming
      是往後 `run_pipeline` 實際會用的路徑，重新捕捉反映的是「未來正確
      行為」，不是遮蓋迴歸，`np.array_equal` 逐位元組相等的比對標準
      維持不變，只換了這一個 golden 檔案內容，另外兩個 golden 檔案
      （byte-for-byte 驗證過未被觸碰）維持原樣。
  - [x] **`compose_global_transforms` 記憶體待辦的後續調查——部分解決，
    不是完全解決，過程中意外發現一個獨立的 pose-graph 退化案例**：
    上一輪發現 `compose_global_transforms` 單獨佔 `run_pipeline` 總
    peak 的 97%（N=100，15,948KB）之後，懷疑根因是稠密 Jacobian 隨
    pose-graph 規模增長，這一輪做了完整的根因驗證，過程與結論如下：
    - **稀疏 Jacobian 方向：已評估並推翻**。用真實的殘差結構（每個
      node 4 個自由度 `a,b,tx,ty`，不是原本猜的 6 個；每條 edge 6 個
      殘差、依賴 src/dst 兩個 node 共 8 個變數；每個 GPSAnchor/
      YawAnchor 2 個殘差、依賴自己 node 的 4 個變數）精算：N=100、
      99 條邊、1 個 GPS anchor（這批合成測資實際只有 1 個，不是每個
      node 都有）時，稠密 Jacobian 只有 ≈1.9MB，對照 15,948KB 的實測
      總 peak，即使把 Jacobian 完全消除也只省 ≈12%——落在「邊際改善」
      的範圍，稀疏化不值得投入，根因在別的地方。
    - **真正的根因：scipy 預設用數值微分（有限差分）估計 Jacobian，
      每算一次要呼叫 `residuals()`「參數量+1」次，且每次外層疊代都
      重算一次**。實測驗證：N=100 時 `residuals()` 總共被呼叫 3,144
      次，不是 `result.nfev` 顯示的 8 次（那個欄位算的是外層疊代數，
      不是原始函式呼叫數）——3144 = 8 疊代 × (392+1) 參數，精確吻合，
      而且不是巧合：檢查了呼叫模式，每組 393 次呼叫裡，392 次都跟
      該組基準點恰好差 1 個座標，是有限差分數值微分的明確特徵。
    - **修法：手推封閉形式的解析 Jacobian，取代數值微分**。因為
      `residuals()` 裡每一項（edge 的旋轉/平移子項、GPSAnchor、
      YawAnchor）對決策變數都是仿射（沒有任何兩個決策變數的乘積），
      偏微分有解析封閉形式，而且這個 Jacobian 是「常數矩陣」——不管
      在哪個參數點估計都一樣（這個結論本身也已經數值驗證過：4 個
      獨立參數點，包含一個刻意測試的近退化案例，跟 scipy 有限差分的
      結果逐元素比對，最大誤差 ~1e-9，落在有限差分本身的雜訊範圍內）。
      驗證方法：手推兩次（edge 項用了兩種獨立推導路徑互相驗證：直接
      對展開式微分、以及用 `R=a·I+b·J`（I=單位陣，J=90°旋轉生成元）
      的結構性線性分解重新推一次，逐項吻合）+ sympy 符號運算獨立驗證
      edge 項 + 數值交叉驗證（`scipy.optimize._numdiff.approx_derivative`
      對照，4 個點）。實作為 `posegraph.py` 的 `_analytic_jacobian`/
      `_edge_diff_jacobian_blocks`，接進 `least_squares` 的 `jac=`
      參數（scipy 的 `jac` 只接受 callable，不接受固定矩陣——但因為
      這個矩陣本身是常數，callable 直接忽略傳入的參數、回傳同一個
      預先算好的矩陣即可，不需要 memoization/cache，是很自然的寫法）。
    - **效果：呼叫次數精確消除，但記憶體只降了一部分，還沒解決**。
      `residuals()` 呼叫次數從 3,144 精確降到 8（跟疊代次數 1:1，
      `njev==nfev==8`）——這個部分完全達成預期。但 `compose_global_
      transforms` 的記憶體 peak 只從 15,948KB 降到 13,411KB
      （≈16%），遠低於呼叫次數消除的幅度（393 倍 vs 16%）。**這代表
      「呼叫次數放大」雖然是真實存在、也被正確識別的問題，但不是
      記憶體 peak 的主要來源**：`tracemalloc` 的 peak 是某個瞬間的
      最高同時用量，不是所有呼叫的暫態配置總和，如果每次 `residuals()`
      呼叫自己的暫態配置都有正常釋放、不會跨呼叫累積，呼叫次數再多
      也不該讓「同一瞬間」的用量變大很多。真正的主導成本比較可能在
      `scipy.optimize.least_squares`（`method='trf'`）內部——診斷時
      在 `scipy/optimize/_differentiable_functions.py:754` 附近看到
      呼叫結束後還留著約 1.8MB 未釋放的配置，但這只是一條線索，還沒
      追到底（可能是信賴域子問題求解過程中的 QR/SVD workspace，或
      其他 TRF 內部結構，都還沒驗證）。**這個待辦保持開放，10 月正式
      上大規模資料前仍需要進一步調查**，跟稀疏 Jacobian 一樣，不要
      未經驗證就投入下一個「聽起來合理」的方向。
    - **意外發現、需要獨立記錄的一件事：導入解析 Jacobian 後，某個
      既有 golden-fixture regression 測試（`isolated_node_without_
      gps_anchor`）的輸出從 (93,70,3) 變成 (36,22,3)，一度看起來像是
      解析 Jacobian 推導錯誤，但追查後確認不是**——這個測試情境裡，
      node 1→2 與 2→3 的邊都刻意設計成失敗（測「孤立節點」的分類
      邏輯），而這個情境沒有提供任何 GPS 座標，導致 node 3、4 組成
      一個完全跟主圖（以 node 0 為錨點）斷開、沒有任何 anchor 的
      子圖。驗證：node 3 對 edge(3,4) 約束的殘差在新結果裡 ≈1e-10
      （完全收斂、是合法的局部最優解），但 node 3/4 的絕對姿態
      （scale≈4.68、旋轉≈244°）跟舊結果（scale≈1、旋轉≈0°）完全
      不同——這是一個真實存在、沿著「整個子圖一起做任意剛體變換
      不改變彼此的邊殘差」這個方向的平坦/退化方向，數值路徑的極小
      差異（有限差分 vs 解析）剛好落在這個平坦方向的不同點上，兩個
      解都是合法局部最優解，不是誰對誰錯。這是 pose graph 退化家族
      裡的**另一種顯化形式**，跟上面已經記錄的大規模 GPS-anchor
      耦合造成的 scale 崩潰是同一類根因（結構性缺乏約束，答案對數值
      路徑敏感），不是解析 Jacobian 引入的新問題——已用 4 種獨立方法
      驗證過 Jacobian 推導本身正確（兩種手推、sympy、數值交叉驗證）。
      **處理方式：不重新捕捉這個 golden、不 revert 解析 Jacobian，
      改成修正測試 fixture 本身的拓樸**——`_isolated_node_scenario`
      新增一條 `(1,3)` bypass 邊（跳過孤立的 node 2，讓 node 3/4
      重新連回錨定的主圖），因為 golden-fixture regression 測試的
      前提是「輸出應該唯一、可比對」，這個前提在斷開子圖的拓樸下
      從一開始就不成立，繼續拿它當回歸基準是在測一個天生不穩定的
      東西，不是這次改動造成的。修正後驗證過：新舊 Jacobian 在修正
      拓樸下收斂到完全相同的 canvas shape (11,16,3)，既有分類斷言
      （node 2 仍然失敗、其餘成功）也全部不受影響。`all_images_well_
      matched` 那條 golden 的 1 像素 canvas 差異則維持先前「streaming
      正確行為」同一套處理方式：重新捕捉，因為兩個結果的
      `residual_error` 都在 machine-precision 等級（≈1e-16），只是一個
      接近零的 `b` 參數正負號差異，剛好卡在 canvas 尺寸的整數進位
      邊界上，不是真實的解不同。
- [ ] FastAPI
