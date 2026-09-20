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

## 目前狀態
- [x] SSH + VS Code Remote-SSH + Claude Code CLI 環境
- [x] 專案骨架
- [x] metrics.py + unit tests
- [x] EXIF/XMP 解析 (GPS 座標讀取 + 局部平面投影，21/21 tests passing)
- [ ] direct georeferencing (geo/camera.py, geo/direct.py) 仍暫緩，見「已知的暫緩事項」
- [ ] feature-based pipeline
- [ ] FastAPI
