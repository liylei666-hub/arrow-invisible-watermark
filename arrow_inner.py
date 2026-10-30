"""ARROW 隐形水印 —— 换载体实验：以「内孔轮廓」为信息载体。

与 `arrow_wm.py`（边缘宽度调制）的区别只有一处：信息写在哪里。

    边缘版：调制字母**外轮廓**的笔画宽度 -> 外轮廓出现 ±1.8px 波浪
    内孔版：外轮廓**一个像素都不动**，只调制 4 个内孔（O 内环、两个 R 字怀、
            A 上三角）的边界半径 -> 波浪被关进字母内部

容量：4 个内孔周长合计 2525 px，140 个码元 -> 平均每格 18 px。

复用 `arrow_wm` 的：码字编码（Hamming 7,4）、仿真器、配准/矫正链路。
"""
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import arrow_wm as A

DEPTH: float = 2.2          # 内孔边界径向峰值位移（px @ 底版尺度，约 0.022 mm）
# 实测（n=20，heavy 档）：1.8->17/20, 2.2->18/20, 2.5->17/20, 3.0->19/20。
# 取 2.2 是因为内孔本身只有 0.1~0.3 mm，这个幅度在实物尺寸下仍不可辨。
N_THETA: int = 1440         # 每个内孔的圆周采样数
SCAN_HALF: float = 10.0     # 半径搜索半宽（px）
SCAN_STEP: float = 0.25     # 半径搜索步长（px）


# ==============================================================================
# 1. 载体：内孔轮廓
# ==============================================================================
class InnerCarrier:
    """提取内孔、按周长分配码元、把 70 bit 写成内孔边界的径向微波。"""

    def __init__(self, design: Optional[A.ArrowDesign] = None,
                 depth: float = DEPTH, verbose: bool = False):
        self.design = design or A.ArrowDesign()
        self.depth = float(depth)
        self.SS = A.SS
        self.theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
        self.holes = self._extract_holes()
        self._alloc_cells()
        if verbose:
            tot = sum(h["peri"] for h in self.holes)
            print(f"[InnerCarrier] {len(self.holes)} 个内孔，总周长 {tot:.0f} px @1x")
            for i, h in enumerate(self.holes):
                print(f"  #{i} 周长 {h['peri'] / self.SS:6.1f}px  "
                      f"{h['n_cells']:2d} 码元  "
                      f"每格 {h['peri'] / self.SS / h['n_cells']:4.1f}px  "
                      f"圆心 ({h['c'][0] / self.SS:.0f},{h['c'][1] / self.SS:.0f})")

    # ---------------- 内孔提取 ----------------
    def _extract_holes(self) -> List[Dict]:
        ink = (self.design.mask4 > 127).astype(np.uint8)
        bg = (1 - ink).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bg, 4)
        H, W = ink.shape
        out: List[Dict] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if x == 0 or y == 0 or x + w >= W or y + h >= H:
                continue                     # 与画幅相连 -> 外部背景，不是孔
            if area < 800:
                continue                     # 4x 域，约等于 1x 的 50 px^2
            m = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = cnts[0].reshape(-1, 2).astype(np.float64)
            peri4 = float(cv2.arcLength(cnts[0], True))
            cen = c.mean(axis=0)
            d = c - cen
            r = np.hypot(d[:, 0], d[:, 1])
            th = np.arctan2(d[:, 1], d[:, 0]) % (2.0 * np.pi)
            order = np.argsort(th)
            th_s, r_s = th[order], r[order]
            # 周期延拓后线性插值 -> 稠密的 r(theta) 表
            th_ext = np.concatenate([th_s - 2 * np.pi, th_s, th_s + 2 * np.pi])
            r_ext = np.tile(r_s, 3)
            r_tab = np.interp(self.theta, th_ext, r_ext)
            out.append(dict(c=cen, r_tab=r_tab, peri=peri4,
                            poly_orig=np.round(c).astype(np.int32)))
        # 按周长降序，保证分配稳定
        out.sort(key=lambda h: -h["peri"])
        return out

    # ---------------- 码元分配 ----------------
    def _alloc_cells(self) -> None:
        n_tot = A.N_CELLS
        per = np.array([h["peri"] for h in self.holes], np.float64)
        raw = per / per.sum() * n_tot
        alloc = np.maximum(2, (np.floor(raw / 2.0) * 2).astype(int))
        # 用最大余数法补齐到 n_tot（每次 +2）
        while alloc.sum() < n_tot:
            score = raw - alloc
            alloc[int(np.argmax(score))] += 2
        while alloc.sum() > n_tot:
            cand = [i for i in range(len(alloc)) if alloc[i] > 2]
            if not cand:
                break
            score = np.array([alloc[i] - raw[i] for i in range(len(alloc))])
            for i in range(len(alloc)):
                if alloc[i] <= 2:
                    score[i] = -1e9
            alloc[int(np.argmax(score))] -= 2
        for h, k in zip(self.holes, alloc):
            h["n_cells"] = int(k)
            h["dtheta"] = 2.0 * np.pi / k

        # 全局码元表：先孔、后角度
        self.cells: List[Tuple[int, int]] = []
        for hi, h in enumerate(self.holes):
            for k in range(h["n_cells"]):
                self.cells.append((hi, k))
        assert len(self.cells) == n_tot, f"码元数 {len(self.cells)} != {n_tot}"

    # ---------------- 每个码元的平滑窗 ----------------
    def _window(self, hi: int, k: int) -> np.ndarray:
        h = self.holes[hi]
        th = self.theta
        t = ((th - k * h["dtheta"]) % (2.0 * np.pi)) / h["dtheta"]
        inside = t < 1.0
        return np.where(inside, np.sin(np.pi * np.clip(t, 0.0, 1.0)), 0.0)

    def _delta(self, bits: List[int]) -> List[np.ndarray]:
        """返回每个内孔的径向位移场 delta(theta)。"""
        fields = [np.zeros(N_THETA) for _ in self.holes]
        for i, b in enumerate(bits):
            ka, kb = 2 * i, 2 * i + 1
            hi_a, k_a = self.cells[ka]
            hi_b, k_b = self.cells[kb]
            sgn = 1.0 if b == 1 else -1.0
            fields[hi_a] += sgn * self.depth * self._window(hi_a, k_a)
            fields[hi_b] -= sgn * self.depth * self._window(hi_b, k_b)
        return fields

    def solid_mask4(self) -> np.ndarray:
        """把内孔填满后的 4x 掩膜 —— 只保留字母外轮廓。"""
        m = self.design.mask4.copy()
        for h in self.holes:
            cv2.fillPoly(m, [h["poly_orig"]], 255)
        return m

    def solid_template(self) -> np.ndarray:
        return self.design.render(self.solid_mask4())

    # ---------------- 嵌入 ----------------
    def embed(self, digits: str) -> Dict:
        bits = A.encode_payload(digits)
        mask4 = self.design.mask4.copy()
        fields = self._delta(bits)
        for hi, h in enumerate(self.holes):
            # 先把原孔整块填回墨色，再按调制后的多边形重新挖空
            cv2.fillPoly(mask4, [h["poly_orig"]], 255)
            rr = h["r_tab"] + fields[hi] * self.SS
            rr = np.maximum(rr, 4.0)
            pts = np.stack([h["c"][0] + rr * self.cos_t,
                            h["c"][1] + rr * self.sin_t], axis=1)
            cv2.fillPoly(mask4, [np.round(pts).astype(np.int32)], 0)
        plate = self.design.render(mask4)
        return {"digits": digits, "bits": bits, "plate": plate, "mask4": mask4}


# ==============================================================================
# 2. 解码器：内孔边界亚像素半径 + 匹配滤波
# ==============================================================================
class InnerDecoder:
    def __init__(self, carrier: InnerCarrier, use_ecc_refine: bool = True,
                 solid_reg: bool = False):
        self.carrier = carrier
        self.design = carrier.design
        # 复用边缘版的配准/矫正链路（自同步 + ECC 精配 + 粗搜抢救）
        self._reg = A.WatermarkDecoder(self.design, use_ecc_refine=use_ecc_refine)
        if solid_reg:
            # 实测：填孔后 ECC 反而更差（17/20 -> 14/20）。内孔是锐利且分布均匀的
            # 强特征，填掉等于自废配准基准。默认关闭，仅留作对照开关。
            self._use_solid_tmpl()
        self._build_maps()
        # 参考半径用「未调制模板」实测得到：这样 4x 掩膜轮廓与 1x 灰度跳变沿
        # 之间随角度变化的系统偏差会被逐点抵消，只剩真正的调制位移。
        profs = self._profiles(self.design.template.astype(np.float32))
        self._r_ref = [self._radius(p, r0) for p, (_, _, r0) in zip(profs, self._maps)]

    def _use_solid_tmpl(self) -> None:
        r = self._reg
        solid = self.carrier.solid_template().astype(np.float32) / 255.0
        r.tmpl = solid
        th, tw = r._tmpl_half.shape
        r._tmpl_half = cv2.resize(solid, (tw, th), interpolation=cv2.INTER_AREA)
        qs = r._quad_scale
        tq = cv2.resize(solid, (int(round(A.CANVAS_W * qs)),
                                int(round(A.CANVAS_H * qs))),
                        interpolation=cv2.INTER_AREA)
        r._tmpl_q = tq
        v = tq.ravel() - tq.mean()
        r._tmpl_vec = v
        r._tmpl_norm = float(np.linalg.norm(v))

    def _build_maps(self) -> None:
        """预先把「沿法向扫描」的采样坐标建成固定网格（几何与解码无关）。"""
        ss = self.carrier.SS
        n_s = int(round(2 * SCAN_HALF / SCAN_STEP)) + 1
        s_grid = np.linspace(-SCAN_HALF, SCAN_HALF, n_s)
        self._n_s = n_s
        self._s_grid = s_grid
        self._maps: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for h in self.carrier.holes:
            r0 = h["r_tab"] / ss                       # 1x 尺度
            cx, cy = h["c"][0] / ss, h["c"][1] / ss
            rad = r0[:, None] + s_grid[None, :]        # (N_THETA, n_s)
            mx = (cx + rad * self.carrier.cos_t[:, None]).astype(np.float32)
            my = (cy + rad * self.carrier.sin_t[:, None]).astype(np.float32)
            self._maps.append((mx, my, r0))

    def _radius(self, prof: np.ndarray, r0: np.ndarray) -> np.ndarray:
        """沿每条射线求 50% 跳变沿的亚像素半径。"""
        n = prof.shape[0]
        bright = np.median(prof[:, :3], axis=1)
        dark = np.median(prof[:, -3:], axis=1)
        thr = 0.5 * (bright + dark)
        below = prof < thr[:, None]
        has = below.any(axis=1)
        first = np.argmax(below, axis=1)
        i1 = np.clip(first, 0, self._n_s - 1)
        i0 = np.clip(i1 - 1, 0, self._n_s - 1)
        rows = np.arange(n)
        v0 = prof[rows, i0]
        v1 = prof[rows, i1]
        span = v0 - v1
        # 分母设下限：平坡处 v0≈v1，直接除会爆成 1e6 量级
        frac = np.where(span > 2.0, (v0 - thr) / np.maximum(span, 1e-6), 0.5)
        frac = np.clip(frac, 0.0, 1.0)
        r = r0 + self._s_grid[i0] + frac * SCAN_STEP
        return np.where(has, r, r0)

    def _profiles(self, img: np.ndarray) -> List[np.ndarray]:
        out = []
        for mx, my, _ in self._maps:
            out.append(cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE))
        return out

    def measure(self, warped: np.ndarray) -> np.ndarray:
        """返回每个码元的径向位移投影 M（单位 px）。"""
        img = warped.astype(np.float32)
        profs = self._profiles(img)
        feats: List[float] = []
        for hi, (prof, (_, _, r0)) in enumerate(zip(profs, self._maps)):
            r_meas = self._radius(prof, r0)
            delta = r_meas - self._r_ref[hi]
            # 物理上限：单侧位移不可能超过扫描半宽，越界视为测量失败
            delta = np.where(np.abs(delta) <= SCAN_HALF, delta, 0.0)
            for k in range(self.carrier.holes[hi]["n_cells"]):
                w = self.carrier._window(hi, k)
                den = float((w * w).sum())
                feats.append(float((delta * w).sum() / max(den, 1e-6)))
        return np.array(feats, np.float32)

    def decode(self, gray: np.ndarray) -> Dict:
        r = self._reg.rectify(gray)
        if not r["ok"]:
            return {"ok": False, "reason": r.get("reason", "rectify_failed")}
        feat = self.measure(r["warped"])
        bits: List[int] = []
        conf: List[float] = []
        for i in range(A.N_BITS):
            d = float(feat[2 * i] - feat[2 * i + 1])
            bits.append(1 if d > 0 else 0)
            conf.append(d)
        conf_a = np.array(conf, np.float32)
        # 软判决最大似然：硬判决在一组内出现 2 位错时会误纠，软判决能救回
        digits, n_chg = A.decode_payload_soft(conf_a)
        hard, corrected, _ = A.decode_payload(bits)
        return {"ok": True, "digits": digits, "digits_hard": hard, "bits": bits,
                "feat": feat, "conf": conf_a,
                "mean_conf": float(np.mean(np.abs(conf_a))) * 0.5,
                "corrected": n_chg, "ecc": r["ecc"], "warped": r["warped"]}


# ==============================================================================
# 3. 命令行：压测 / 导出印版 / 解码实拍
# ==============================================================================
def run_bench(carrier: InnerCarrier, n: int = 20, seed: int = 6000,
              levels=("clean", "light", "moderate", "heavy", "extreme")) -> None:
    dec = InnerDecoder(carrier)
    sim = A.CeramicCamSimulator()
    print("=" * 78)
    print(f"  内孔载体压测  depth={carrier.depth:g}px  每组 {n} 样本")
    print("=" * 78)
    print(f"| {'档位':<12} | {'成功':>7} | {'BER':>8} | {'平均置信度':>10} | {'耗时':>7} |")
    print("|" + "-" * 14 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 9 + "|")
    for lv in levels:
        oks, bers, confs, ts = 0, [], [], []
        for sd in range(n):
            dg = "1%010d" % ((sd + seed) * 7919 % 10 ** 10)
            wm = carrier.embed(dg)
            tb = np.array(wm["bits"])
            cam = sim.simulate(wm["plate"], lv, np.random.default_rng(seed + sd))
            t0 = __import__("time").time()
            r = dec.decode(cam["image"])
            ts.append(__import__("time").time() - t0)
            b = np.array(r["bits"]) if r["ok"] else np.zeros(A.N_BITS, int)
            bers.append(float(np.mean(b != tb)) * 100.0)
            oks += int(r.get("digits") == dg)
            if r["ok"]:
                confs.append(r["mean_conf"])
        print(f"| {lv:<12} | {oks:>3d}/{n:<3d} | {np.mean(bers):>7.2f}% | "
              f"{np.mean(confs) if confs else 0:>10.4f} | {np.mean(ts):>6.2f}s |")
    print("=" * 78)


def export_plates(carrier: InnerCarrier, out_dir: str,
                  payloads=("13800138000", "13912345678", "18600001234",
                            "01088886666", "17701234567")) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dec = InnerDecoder(carrier)
    A.imwrite_u(os.path.join(out_dir, "ARROW_无码对照.png"), carrier.design.template)
    print(f"{'号码':<14}{'文件':<30}{'自检':>8}{'置信度':>9}")
    print("-" * 64)
    for dg in payloads:
        wm = carrier.embed(dg)
        name = f"ARROW_{dg}.png"
        A.imwrite_u(os.path.join(out_dir, name), wm["plate"])
        A.imwrite_u(os.path.join(out_dir, f"ARROW_{dg}_1bit.png"),
                    np.where(wm["plate"] > 127, 255, 0).astype(np.uint8))
        r = dec.decode(wm["plate"])
        ok = "OK" if r.get("digits") == dg else "FAIL"
        print(f"{dg:<14}{name:<30}{ok:>8}{r.get('mean_conf', 0):>9.4f}")
    print("-" * 64)


def main(argv=None) -> int:
    import argparse
    import os as _os
    ap = argparse.ArgumentParser(description="ARROW 内孔载体隐形水印")
    ap.add_argument("--bench", action="store_true", help="跑扰动压测")
    ap.add_argument("-n", type=int, default=20, help="每档样本数")
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--depth", type=float, default=DEPTH, help="径向峰值位移 (px)")
    ap.add_argument("--export", metavar="DIR", default=None, help="导出送印图到目录")
    ap.add_argument("--decode", nargs="+", metavar="IMG", default=None,
                    help="解码实拍照片")
    args = ap.parse_args(argv)

    carrier = InnerCarrier(depth=args.depth, verbose=True)
    if args.bench:
        run_bench(carrier, n=args.n, seed=args.seed)
    if args.export:
        export_plates(carrier, args.export)
    if args.decode:
        dec = InnerDecoder(carrier)
        print("=" * 70)
        print("  实拍照片解码（内孔载体）")
        print("=" * 70)
        for path in args.decode:
            img = A.imread_u(path)
            if img is None:
                print(f"  {_os.path.basename(path):<30} -> 读不到文件")
                continue
            r = dec.decode(img)
            if r["ok"]:
                tag = "" if r["mean_conf"] >= 0.012 else "   [置信度低，可能未嵌入水印]"
                print(f"  {_os.path.basename(path):<30} -> {r['digits']}   "
                      f"置信度 {r['mean_conf']:.4f}{tag}")
            else:
                print(f"  {_os.path.basename(path):<30} -> 解码失败（{r['reason']}）")
        print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
