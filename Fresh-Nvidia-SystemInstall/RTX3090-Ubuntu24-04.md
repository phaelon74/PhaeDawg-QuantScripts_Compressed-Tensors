# Fresh Ubuntu 24.04 + RTX 3090 Setup Guide

## Purpose

This guide walks through setting up a **brand new, headless Ubuntu 24.04** system with **NVIDIA RTX 3090** GPUs for LLM inference (vLLM / llama.cpp), including:

1. Confirming / enabling **Resizable BAR (ReBAR)** via the correct vBIOS
2. Installing the **P2P-enabled open NVIDIA kernel modules** (`610.43.03-p2p`)
3. Installing **CUDA Toolkit 13.2** (network `.deb` method)
4. Installing and configuring **LACT** for power/clock/fan control
5. Setting up **two separate vLLM environments** (`~/latest-release-vllm` and `~/nightly-vllm`)

**Target assumptions for this guide:**

- Ubuntu 24.04 LTS, headless (CLI only)
- Secure Boot **OFF**
- One or more RTX 3090s
- You will use the stock NVIDIA **610.43.03** userspace `.run` installer with **`--no-kernel-modules`**, then build/install the [aikitoria P2P kernel modules](https://github.com/aikitoria/open-gpu-kernel-modules/tree/610.43.03-p2p)

> **Warning:** Flashing GPU vBIOS can brick a card if you flash the wrong ROM. Always match **exact board / subsystem ID / model**. Prefer manufacturer tools first.

---

## Table of Contents

1. [Identify ReBAR / vBIOS status](#1-identify-rebar--vbios-status)
2. [Find and install the correct ReBAR vBIOS (by AIB)](#2-find-and-install-the-correct-rebar-vbios-by-aib)
3. [BIOS settings to lock in](#3-bios-settings-to-lock-in)
4. [Disable Nouveau + prepare the system](#4-disable-nouveau--prepare-the-system)
5. [GRUB IOMMU flags (Intel and AMD)](#5-grub-iommu-flags-intel-and-amd)
6. [Install NVIDIA 610.43.03 userspace + P2P kernel modules](#6-install-nvidia-6104303-userspace--p2p-kernel-modules)
7. [Install CUDA Toolkit 13.2](#7-install-cuda-toolkit-132)
8. [Pin / hold NVIDIA packages so apt never overwrites drivers](#8-pin--hold-nvidia-packages-so-apt-never-overwrites-drivers)
9. [Install and configure LACT](#9-install-and-configure-lact)
10. [Create vLLM environments](#10-create-vllm-environments)
11. [Quick verification checklist](#11-quick-verification-checklist)

---

## 1. Identify ReBAR / vBIOS status

### 1.1 Identify your exact card(s)

```bash
lspci -nn | grep -i nvidia
sudo lspci -nnv -d 10de: | less
```

Note for each GPU:

- **Device ID** (RTX 3090 is typically `10DE:2204`)
- **Subsystem ID** (vendor-specific, e.g. `103C:88D5`, `1462:....`, `1043:....`)
- Slot / PCI address (e.g. `0000:01:00.0`)

Also write down the printed board name from the sticker on the card (e.g. `TUF-RTX3090-O24G-GAMING`, `GV-N3090GAMING OC-24GD`).

### 1.2 Check whether ReBAR is already enabled

After NVIDIA drivers are installed (later in this guide), the easiest checks are:

```bash
nvidia-smi -q | grep -i -A2 "BAR1"
```

Interpretation:

- **BAR1 Total ≈ 256 MiB** → ReBAR is **not** active (common stock early 3090 firmware)
- **BAR1 Total ≈ 24576 MiB (24 GiB)** → ReBAR is **active**

With LACT installed (later):

```bash
lact cli info | grep -i -E "Resize|VBIOS|Model"
```

You want `Resizeable bar: Enabled` (LACT spelling).

### 1.3 Read current VBIOS version (after drivers)

```bash
nvidia-smi -q | grep -i "VBIOS"
```

Save that string. You will use it to confirm any TechPowerUp ROM is a ReBAR-capable revision for **your** exact board, not a random 3090 ROM.

---

## 2. Find and install the correct ReBAR vBIOS (by AIB)

### Rule of priority

1. **Manufacturer official ReBAR / BIOS update tool first** (always)
2. Only if manufacturer has **no** usable update for your exact SKU: TechPowerUp VGA BIOS database
3. Never flash a ROM from a different board partner, cooler variant, or subsystem ID

### Official / first-stop sources

NVIDIA maintains a partner index here:

- [NVIDIA Resizable BAR Firmware Update Tool](https://nvidia.custhelp.com/app/answers/detail/a_id/5165)
- Overview: [GeForce RTX 30 Series Resizable BAR](https://www.nvidia.com/en-us/geforce/news/geforce-rtx-30-series-resizable-bar-support/)

### AIB / OEM lookup table

| Vendor / AIB | Where to get a solid ReBAR vBIOS | Notes |
|---|---|---|
| **NVIDIA Founders Edition** | Official NVIDIA ReBAR firmware tool from [NVIDIA KB A5165](https://nvidia.custhelp.com/app/answers/detail/a_id/5165) | FE tool is **FE-only**. Do not use on partner cards. |
| **ASUS** (TUF / ROG STRIX / Turbo) | ASUS Support → your exact model → **Driver & Utility / BIOS** (search model on [asus.com](https://www.asus.com/support/)) | Prefer ASUS `RTX3090_Vx.exe` Resizable BAR packages. If the Windows tool wrongly says “no update needed,” extract the package and flash the matching `.rom` with `nvflash` only after verifying board ID. |
| **Gigabyte / AORUS** | Card support page BIOS section, or [Gigabyte Resizable BAR page](https://www.gigabyte.com/WebPage/785/NVIDIA_resizable_bar.html) | BIOS series matter (e.g. F1→F2–F9 only). Wrong series can brick. |
| **MSI** | MSI product support → Utility / Live Update / Dragon Center path for your exact 3090 model | Start from manufacturer; do not grab a random MSI 3090 ROM. |
| **EVGA** | EVGA support / Precision-era BIOS downloads for your exact 3090 P/N | EVGA is shut down as a brand; if official pages are gone, use archived official packages if you still have them, else TechPowerUp **only** with exact P/N + subsystem match. |
| **ZOTAC** | [ZOTAC Get Resizable BAR](https://www.zotac.com/news/get-resizable-bar) / Download Center → Graphics Cards → BIOS | Filter by exact series/model. |
| **PNY** | PNY support page for your exact XLR8 / Blower SKU | If PNY has no ReBAR package for your SKU, search TechPowerUp for that **exact** subsystem ID and confirm ReBAR capability before flashing. |
| **Palit** | Palit product page → Tool / BIOS for that SKU | Manufacturer first. |
| **Gainward** | Gainward product page → Tool for that SKU | Same family as Palit in many cases; still match exact model. |
| **Galax / KFA2** | Galax / KFA2 support → BIOS for exact model | Manufacturer first. |
| **Colorful / Inno3D / Manli / Other AIBs** | Vendor support page for exact model; NVIDIA partner list links from [KB A5165](https://nvidia.custhelp.com/app/answers/detail/a_id/5165) | If no official ReBAR package exists, TechPowerUp only with strict matching. |
| **HP / Dell / Lenovo OEM workstation cards** | OEM support only | Many OEM 3090s never received a public ReBAR vBIOS. If the OEM never published one, **do not invent one** from a desktop AIB ROM. |
| **Unknown / sticker missing** | Identify via `lspci -nn` subsystem ID, then search manufacturer + TechPowerUp by that ID | Do not flash until subsystem ID, board name, and VBIOS family all match. |

### If you must use TechPowerUp

Use: [TechPowerUp VGA BIOS Collection](https://www.techpowerup.com/vgabios/)

Before flashing a TechPowerUp ROM, verify **all** of the following match your live card:

1. Exact **GPU** = GeForce RTX 3090 (`10DE:2204`)
2. Exact **Subsystem ID** (e.g. `XXXX:YYYY`) matches `lspci -nn`
3. Exact board name / cooler / memory vendor notes look right
4. ROM notes / version indicate **Resizable BAR / ReBAR** support (or is known to be a post-ReBAR manufacturer revision)
5. You have a **backup** of the currently running VBIOS

Backup example (after drivers + `nvflash` available):

```bash
sudo ./nvflash --save backup-gpu0.rom
# multi-GPU: target the correct index
sudo ./nvflash --index 1 --save backup-gpu1.rom
```

Only then:

```bash
sudo ./nvflash your-verified-rebar.rom
# or, if your board requires bypassing a board-ID check with a manufacturer-extracted ROM:
# sudo ./nvflash -6 --index N your-verified-rebar.rom
```

Reboot and re-check BAR1 / LACT ReBAR status.

> **Do not proceed with P2P / multi-GPU tuning until ReBAR is confirmed enabled.** BAR1 P2P path on 3090s without NVLink depends on a full BAR window.

---

## 3. BIOS settings to lock in

Enter motherboard firmware setup and set:

| Setting | Value |
|---|---|
| Secure Boot | **Disabled** |
| Above 4G Decoding | **Enabled** |
| Resizable BAR / Re-Size BAR | **Enabled** (or Auto, if that is the only option that sticks) |
| CSM / Legacy Boot | **Disabled** (UEFI only) |
| IOMMU / VT-d / AMD-Vi | **Enabled** (passthrough mode is applied later via GRUB `iommu=pt`) |
| ACS / Access Control Services on PCIe root ports | **Disabled** if the option exists (ACS kills P2P bandwidth) |
| Primary display / iGPU | Prefer iGPU or whatever lets the box boot headless reliably |

Save & Exit.

---

## 4. Disable Nouveau + prepare the system

SSH into the fresh Ubuntu 24.04 install (or use local console).

### 4.1 Update base system and install build deps

```bash
sudo apt update
sudo apt -y upgrade
sudo apt -y install build-essential git curl wget ca-certificates \
  linux-headers-$(uname -r) pkg-config libglvnd-dev
```

> We intentionally do **not** install/use DKMS for NVIDIA. The P2P open modules are built and installed manually.

### 4.2 Blacklist Nouveau

```bash
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot
```

After reboot, confirm Nouveau is gone:

```bash
lsmod | grep -i nouveau || echo "nouveau not loaded (good)"
```

---

## 5. GRUB IOMMU flags (Intel and AMD)

P2P DMA requires IOMMU **passthrough** (`iommu=pt`), not full translation.

Edit:

```bash
sudo nano /etc/default/grub
```

Find `GRUB_CMDLINE_LINUX_DEFAULT` and add the flags for your CPU vendor.

### AMD

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=on iommu=pt"
```

### Intel

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt"
```

Optional (only if ACS cannot be disabled in BIOS and your kernel supports it):

```text
pcie_acs_override=downstream,multifunction
```

Apply and reboot:

```bash
sudo update-grub
sudo reboot
```

Verify after reboot:

```bash
cat /proc/cmdline
dmesg | grep -i iommu | head
```

You should see your `iommu=pt` flag present.

---

## 6. Install NVIDIA 610.43.03 userspace + P2P kernel modules

Confirmed intended flow:

1. Install stock **610.43.03** `.run` with **`--no-kernel-modules`**
2. Clone/build/install [aikitoria `610.43.03-p2p`](https://github.com/aikitoria/open-gpu-kernel-modules/tree/610.43.03-p2p) via `./install.sh`
3. Reboot

### 6.1 Download the official 610.43.03 package runner

Driver page: [NVIDIA Linux 610.43.03](https://www.nvidia.com/en-us/drivers/details/274183/)

Direct package URL pattern:

```bash
cd ~
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/610.43.03/NVIDIA-Linux-x86_64-610.43.03.run
chmod +x NVIDIA-Linux-x86_64-610.43.03.run
```

### 6.2 Purge any distro NVIDIA packages (if present)

```bash
sudo apt -y purge 'nvidia-*' 'libnvidia-*' || true
sudo apt -y autoremove
```

### 6.3 Run the `.run` installer (userspace only — NO kernel modules / NO DKMS)

```bash
sudo ./NVIDIA-Linux-x86_64-610.43.03.run --no-kernel-modules
```

#### Prompt guidance (headless LLM box)

| Prompt / decision | Choose | Why |
|---|---|---|
| Install 32-bit compatibility libraries? | **No** | Not needed for vLLM / llama.cpp 64-bit inference. |
| Install NVIDIA's kernel module / register with DKMS? | **No** (and you already passed `--no-kernel-modules`) | Kernel modules come from the P2P open-source tree instead. |
| Rebuild initramfs? | **Yes**, if asked | Keeps Nouveau blacklisted and driver/userspace consistent across boots. |
| Run `nvidia-xconfig` / configure Xorg? | **No** | Headless box; CLI only. |
| Overwrite libglvnd files? | Accept NVIDIA defaults if prompted | Needed for a clean userspace install even without a desktop. |

If the installer offers an expert/advanced path, stay on the simple path and keep kernel modules disabled.

> **Note on initramfs:** Even with `--no-kernel-modules`, say **Yes** to rebuilding initramfs if prompted. Separately, you already ran `update-initramfs -u` after blacklisting Nouveau. After the P2P modules are installed, `./install.sh` runs `depmod`; a final reboot loads the new modules cleanly. If you ever manually change module blacklists again, re-run `sudo update-initramfs -u`.

### 6.4 Clone and install the P2P open kernel modules

```bash
cd ~
git clone --branch 610.43.03-p2p --depth 1 \
  https://github.com/aikitoria/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules
./install.sh
```

What `install.sh` does:

```bash
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia
make modules -j$(nproc)
sudo make modules_install -j$(nproc)
sudo depmod
nvidia-smi
```

If modules were not loaded yet (fresh install), the `rmmod` lines may warn; that is fine. `nvidia-smi` should work after modules install; if not, reboot:

```bash
sudo reboot
```

### 6.5 Verify driver + ReBAR + multi-GPU visibility

```bash
nvidia-smi
nvidia-smi -q | grep -i -E "Driver Version|VBIOS|BAR1"
cat /proc/cmdline
```

Expected:

- Driver **610.43.03**
- All GPUs visible
- BAR1 ≈ full VRAM if ReBAR is enabled

Optional P2P sanity (after CUDA samples / toolkit are available): use NVIDIA `p2pBandwidthLatencyTest` and confirm P2P-enabled bandwidth is higher than P2P-disabled copies. Slow P2P usually means IOMMU not in `pt` mode and/or ACS still enabled.

---

## 7. Install CUDA Toolkit 13.2

Use the **Ubuntu 24.04 network deb** method from NVIDIA:

- [CUDA 13.2 download archive (Ubuntu 24.04, deb network)](https://developer.nvidia.com/cuda-13-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network)

Exact commands:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-2
```

### 7.1 Make `nvcc` available in every new shell

Append CUDA 13.2 to your shell profile:

```bash
cat <<'EOF' >> ~/.bashrc

# CUDA 13.2
export PATH=/usr/local/cuda-13.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF

source ~/.bashrc
nvcc --version
```

You should see CUDA 13.2 without manually exporting paths each session.

> Install **only** `cuda-toolkit-13-2` here. Do **not** install the full `cuda` / `cuda-drivers` metapackages — those can try to pull conflicting NVIDIA driver packages over your P2P stack.

---

## 8. Pin / hold NVIDIA packages so apt never overwrites drivers

Your P2P stack is intentionally outside apt. Protect it.

### 8.1 Apt preferences pin (blocks install/upgrade of driver packages)

```bash
sudo tee /etc/apt/preferences.d/nvidia-driver-block <<'EOF'
Package: nvidia-driver* nvidia-dkms* nvidia-kernel* nvidia-utils* libnvidia* cuda-drivers* nvidia-compute-utils* nvidia-persistenced* nvidia-firmware* nvidia-modprobe* nvidia-settings* xserver-xorg-video-nvidia*
Pin: release *
Pin-Priority: -1
EOF
```

### 8.2 Also mark common packages on hold (belt and suspenders)

```bash
sudo apt-mark hold \
  nvidia-driver-610 nvidia-dkms-610 nvidia-kernel-source-610 \
  nvidia-kernel-common-610 nvidia-utils-610 libnvidia-compute-610 \
  cuda-drivers cuda-drivers-610 2>/dev/null || true
```

### 8.3 Verify apt cannot pull drivers

```bash
apt-cache policy nvidia-driver-610 cuda-drivers | sed -n '1,40p'
sudo apt-get -s upgrade | grep -i nvidia || echo "No nvidia packages in simulated upgrade (good)"
```

From now on, normal `apt upgrade` / `dist-upgrade` must **not** replace your `.run` userspace or P2P kernel modules.

If you later intentionally update the P2P stack, do it manually (new matching `.run` + matching open-gpu-kernel-modules branch), not via apt.

---

## 9. Install and configure LACT

Use the settings in this repo’s example config:

- [`Fresh-Nvidia-SystemInstall/LACT/config.yaml`](./LACT/config.yaml)

Those power/clock/fan values are the **recommended best settings** for RTX 3090 inference boxes in this workflow (270W power cap, core offset +100, mem offset +1000, fan curve).

### 9.1 Install build dependencies + Rust (headless build)

LACT headless build needs Rust + a few libs. GUI/GTK deps are not required if you build headless.

```bash
sudo apt -y install git make clang pkg-config libdrm-dev hwdata curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
```

### 9.2 Clone and install LACT from source (headless)

```bash
cd ~
git clone https://github.com/ilya-zlobintsev/LACT.git
cd LACT
make build-release-headless
sudo make install
```

This installs `lact` / `lactd` and the systemd unit.

### 9.3 Discover your GPU IDs (manual first run)

Start the daemon once:

```bash
sudo systemctl start lactd
lact cli list-gpus
lact cli info
```

Copy each GPU id string exactly. Example format:

```text
10DE:2204-103C:88D5-0000:01:00.0
10DE:2204-196E:136A-0000:46:00.0
```

Stop the daemon before editing config:

```bash
sudo systemctl stop lactd
```

### 9.4 Create `/etc/lact/config.yaml` with the best settings

```bash
sudo mkdir -p /etc/lact
sudo nano /etc/lact/config.yaml
```

Paste the following template, then **replace the two GPU id keys** with the IDs from `lact cli list-gpus`. Keep the numeric settings as-is (these are the best settings from this repo).

```yaml
version: 5
daemon:
  log_level: info
  admin_group: sudo
  disable_clocks_cleanup: false
apply_settings_timer: 5
gpus:
  'REPLACE_WITH_GPU0_ID':
    fan_control_enabled: true
    fan_control_settings:
      mode: curve
      static_speed: 0.5
      temperature_key: edge
      interval_ms: 500
      curve:
        40: 0.30
        50: 0.40
        60: 0.55
        70: 0.70
        80: 0.90
      spindown_delay_ms: 3000
      change_threshold: 2
      auto_threshold: 40
    power_cap: 270.0
    min_core_clock: 210
    max_core_clock: 1500
    gpu_clock_offsets:
      0: 100
    mem_clock_offsets:
      0: 1000
  'REPLACE_WITH_GPU1_ID':
    fan_control_enabled: true
    fan_control_settings:
      mode: curve
      static_speed: 0.5
      temperature_key: edge
      interval_ms: 500
      curve:
        40: 0.30
        50: 0.40
        60: 0.55
        70: 0.70
        80: 0.90
      spindown_delay_ms: 3000
      change_threshold: 2
      auto_threshold: 40
    power_cap: 270.0
    min_core_clock: 210
    max_core_clock: 1500
    gpu_clock_offsets:
      0: 100
    mem_clock_offsets:
      0: 1000
```

If you have only one GPU, delete the second GPU block. If you have more than two, duplicate a block per GPU id.

You can also start from the repo file and swap IDs:

```bash
# from your clone of this QuantScripts repo, if present on the machine:
sudo cp /path/to/PhaeDawg-QuantScripts_Compressed-Tensors/Fresh-Nvidia-SystemInstall/LACT/config.yaml /etc/lact/config.yaml
sudo nano /etc/lact/config.yaml
```

### 9.5 Enable LACTD on boot and verify wattage changed

```bash
sudo systemctl enable --now lactd
systemctl status lactd --no-pager
nvidia-smi
```

**Easiest confirmation it applied:** `nvidia-smi` power limit / reported power behavior should reflect the **270W** cap instead of the stock ~350W card limit.

```bash
nvidia-smi -q -d POWER | grep -i -E "Power Limit|Current Power|Power Management"
```

Troubleshoot with:

```bash
journalctl -u lactd -b --no-pager | less
lact cli info
```

---

## 10. Create vLLM environments

Use **two separate folders** in your home directory:

- `~/latest-release-vllm` — released PyPI vLLM
- `~/nightly-vllm` — git checkout editable install

Use Ubuntu 24.04’s default Python (no special version pinning).

### 10.1 Install `uv` once

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

### 10.2 Latest release vLLM

```bash
mkdir -p ~/latest-release-vllm
cd ~/latest-release-vllm
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install vllm
python -c "import vllm; print(vllm.__version__)"
deactivate
```

### 10.3 Nightly / editable vLLM (precompiled wheels)

```bash
mkdir -p ~/nightly-vllm
cd ~/nightly-vllm
uv venv
source .venv/bin/activate
uv pip install --upgrade pip

git clone https://github.com/vllm-project/vllm.git
cd vllm

# Use prebuilt wheels / avoid compiling CUDA kernels locally
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto

python -c "import vllm; print(vllm.__version__)"
deactivate
```

### 10.4 Day-to-day activation

Released:

```bash
source ~/latest-release-vllm/.venv/bin/activate
```

Nightly:

```bash
source ~/nightly-vllm/.venv/bin/activate
```

Do not mix the two environments in one shell session.

---

## 11. Quick verification checklist

Run through this once the box is built:

```bash
# Firmware / boot
cat /proc/cmdline | grep -E 'iommu=pt|(amd|intel)_iommu=on'

# Driver
nvidia-smi
nvidia-smi -q | grep -i -E 'Driver Version|BAR1|VBIOS'

# CUDA
nvcc --version

# Apt should not manage nvidia drivers
sudo apt-get -s upgrade | grep -i nvidia || echo "no nvidia apt upgrades pending"

# LACT applied
systemctl is-enabled lactd
nvidia-smi -q -d POWER | grep -i 'Power Limit'

# vLLM envs exist
ls ~/latest-release-vllm/.venv
ls ~/nightly-vllm/.venv
ls ~/nightly-vllm/vllm
```

---

## Reference links

- P2P modules: [aikitoria/open-gpu-kernel-modules `610.43.03-p2p`](https://github.com/aikitoria/open-gpu-kernel-modules/tree/610.43.03-p2p)
- NVIDIA 610.43.03 `.run`: [driver details 274183](https://www.nvidia.com/en-us/drivers/details/274183/)
- CUDA 13.2 network install: [CUDA 13.2.0 download archive](https://developer.nvidia.com/cuda-13-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network)
- ReBAR partner index: [NVIDIA KB A5165](https://nvidia.custhelp.com/app/answers/detail/a_id/5165)
- TechPowerUp VGA BIOS DB: [techpowerup.com/vgabios](https://www.techpowerup.com/vgabios/)
- LACT: [ilya-zlobintsev/LACT](https://github.com/ilya-zlobintsev/LACT)
- Example LACT config in this repo: [`LACT/config.yaml`](./LACT/config.yaml)

---

## Safety reminders

- Flash only a vBIOS that matches your **exact** board / subsystem ID.
- Keep Secure Boot off for this stack.
- Keep apt from managing NVIDIA drivers after the P2P install.
- LACT settings above are intentional undervolt / power-limit values for sustained inference; confirm thermals after first long run.
