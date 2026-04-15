# Detectron2 + VideoPose3D GUI Pipeline 使用指南

這是一個整合了 [Detectron2](https://github.com/facebookresearch/detectron2) 與 [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) 的圖形化介面 (GUI) 工具，旨在簡化從 2D 關鍵點擷取到 3D 骨架重建的繁瑣流程。

本專案包含兩個主要執行檔：
- `d2_p3d_gui_windows.py`: 專為 **Windows** 系統優化，包含自動化環境建立功能與依賴項修復。
- `d2_p3d_gui.py`: 適用於 **Linux** 系統。

---

## ⚠️ 必要前置條件 (Prerequisites)

在執行任何腳本之前，您的系統必須準備好以下核心套件：

1. **安裝 Visual C++ 可轉散發套件 (Windows 必備)**: 
   Windows 系統在執行 PyTorch 與底層 C++ 函式庫時，需要 64 位元的微軟運行庫。
   - 請至微軟官網下載：**[vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)**
   - 執行安裝程式（若系統已有安裝，請點選「修復」以確保檔案完整無損）。
   - 安裝完成後，**強烈建議重新啟動電腦**。
   
2. **安裝 FFmpeg (處理影片與渲染輸出)**: 
   - **Windows**: 請前往 [FFmpeg-Builds Releases (BtbN)](https://github.com/BtbN/FFmpeg-Builds/releases) 下載適用版本（例如 `...-win64-gpl.zip`）。解壓縮後，將 `bin` 資料夾的路徑（例如 `C:\ffmpeg\bin`）加入系統環境變數的 `Path` 中。
   - **Linux**: 可透過終端機執行 `sudo apt update && sudo apt install ffmpeg` 安裝。
   
3. **驗證安裝**: 打開終端機輸入 `ffmpeg -version`，確認可正常執行無報錯。

---

## 📂 專案目錄配置

在使用 GUI 之前，請先將專案 Clone 到同一個父目錄下。假設您的主目錄為 `D:/d2_p3d`，請確保結構如下：

```text
D:/d2_p3d/
├── detectron2/
├── VideoPose3D/
├── venv/               # 虛擬環境目錄 (由 GUI 建立或自行建立)
├── d2_p3d_gui_windows.py
└── d2_p3d_gui.py
```

---

## 🪟 Windows 系統使用說明

### 1. 啟動程式
```cmd
python d2_p3d_gui_windows.py
```

### 2. 設定共享路徑 (SHARED PATHS)
一開始請務必在左側面板設定正確的路徑：
- **Detectron2 root dir**: 選擇 Clone 下來的目錄 (例如 `D:/d2_p3d/detectron2`)。
- **VideoPose3D root dir**: 選擇 Clone 下來的目錄 (例如 `D:/d2_p3d/VideoPose3D`)。
- **Venv dir (shared)**: 選擇或輸入虛擬環境路徑 (例如 `D:/d2_p3d/venv`)。

### 3. 環境初始化 (ENVIRONMENT SETUP)
- **選擇 CUDA 版本**: 根據您的顯卡驅動選擇對應版本（可以用指令 `nvidia-smi` 查看支援版本）。
- **點擊 `[+] Create Venv`**: 程式會在指定目錄建立虛擬環境。
- **點擊 `[^] Install Packages`**: 自動安裝 PyTorch、CUDA 依賴項及 Detectron2 預編譯包。

### 4. 執行 Pipeline
- 進入 **Detectron2 頁籤**：選擇圖片或影片，點擊 `RUN ALL` 生成 2D 關鍵點。
- 進入 **VideoPose3D 頁籤**：設定輸出檔名，點擊 `RUN ALL` 進行 3D 推理與渲染。

---

## 🐧 Linux 系統使用說明

### 1. 準備環境
建議先手動建立虛擬環境並安裝基礎套件 (範例為 Conda)：
```bash
conda create -n p3d python=3.10 -y
conda activate p3d
```

### 2. 啟動程式與設定
```bash
python3 d2_p3d_gui.py
```
在 GUI 介面中選擇：
- **Detectron2 / VideoPose3D 目錄**: 指向對應的 Clone 資料夾。
- **Python executable**: 指向虛擬環境內的 Python (例如 `~/miniconda3/envs/p3d/bin/python`)。

### 3. 執行流程
- **Detectron2**: Step 1 會自動以編譯模式 (`pip install -e .`) 安裝。
- **VideoPose3D**: 程式會自動進行檔案修補 (Patch)，解決 Numpy 版本相容性與畫格率 (FPS) 問題。
- 點擊各頁籤的 `RUN ALL` 即可完成轉換。

---

## 🛠️ 常見問題與疑難排解 (Windows 專區)

在 Windows 上執行 PyTorch 與 Detectron2 時，最常遇到底層 DLL 載入失敗的問題。若您在執行時遇到 **`OSError: [WinError 1114] 動態連結程式庫 (DLL) 初始化例行程序失敗。Error loading "...\c10.dll"`**，請依序排查以下幾點：

### 1. 檢查 Python 是否為 Microsoft Store (微軟商店) 版本
微軟商店版的 Python 運行於 `WindowsApps` 沙盒環境中，這會嚴格限制程式跨越目錄去讀取系統底層的硬體 GPU 驅動，導致 `c10.dll` 初始化時被系統阻擋而崩潰。
* **如何檢查**：
    打開 PowerShell，輸入 `where.exe python`。
    *(⚠️ 注意：在 PowerShell 中千萬不要只打 `where python`，因為 `where` 是 `Where-Object` 的縮寫，會吃掉參數。)*
* **解法**：如果輸出的路徑包含 `WindowsApps`，請務必前往 [Python 官方網站](https://www.python.org/downloads/windows/) 下載安裝檔，安裝時勾選「Add Python to PATH」。接著刪除舊的 `venv` 資料夾，用官方版 Python 重新建立虛擬環境。

### 2. Visual C++ 運行庫架構錯誤 (x86 vs x64)
PyTorch 的底層 C++ 函式庫需要 64 位元的微軟運行庫。如果您系統中只有安裝 32 位元 `(x86)` 的版本，初始化將會失敗。
* **檢查與解法**：開啟「新增或移除程式」，確認是否安裝了 **Microsoft Visual C++ 2015-2022 Redistributable (x64)**。若無，請參考前置條件中的連結下載並安裝，安裝後請**重新開機**。

### 3. OpenMP 函式庫衝突 (`libiomp5md.dll`)
如果環境中有其他套件 (如舊版 Numpy) 攜帶了不同版本的 OpenMP DLL，會導致 PyTorch 載入衝突。
* **解法**：
    在虛擬環境中手動補齊 Intel OpenMP：
    ```cmd
    pip install intel-openmp
    ```
    並在您的 Python 腳本最上方加入以下環境變數宣告以壓制衝突：
    ```python
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    ```

### 4. 筆記型電腦的省電阻擋 (雙顯卡)
Windows 的 Optimus 省電機制可能會在 Python 嘗試喚醒獨立顯示卡時將其阻斷。
* **解法**：進入 Windows「設定」>「系統」>「顯示器」>「圖形設定」，將您的虛擬環境 Python 執行檔 (`venv\Scripts\python.exe`) 新增至清單，並強制設為**「高效能」**。