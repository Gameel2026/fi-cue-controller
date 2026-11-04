"""Run the Cortex-M4F build of the frozen Freeze Index cue controller (flash.bin) in the Unicorn ARM emulator.
1) cue-decision agreement with the Python controller on development recordings (frozen D3 and D4, lab or home values)
2) executed instructions, approximate cycles and memory per decision
Run from the mcu folder inside fi_paper:  python run_mcu.py      (needs: pip install unicorn capstone)
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from unicorn import Uc, UcError, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS, UC_HOOK_CODE, UC_HOOK_MEM_READ
from unicorn import arm_const as A

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)
import fi_config as C
import fi_core as K
import ref_ctrl as R

FLASH, RAM, IN, OUT, CUE, PAR = 0x00000000, 0x20000000, 0x60000000, 0x70000000, 0x78000000, 0x7F000000
STACK_TOP = RAM + 0x10000
STOP = 0x0003FF00
CLOCK_HZ = 64e6
DECISIONS_PER_S = 4
N_PER_DATASET = 10
CONTEXT = {"tdcsfog": "lab", "daphnet": "lab", "defog": "home"}
DETS = [("D3_FI_gate_smooth_hyst", False), ("D4_D3_plus_arrest", True)]
TEST_SETS = [("prepared_fi_test", "kaggle_test_tdcsfog", "lab"),
             ("prepared_fi_test", "kaggle_test_defog", "home"),
             ("prepared_fi_external", "fogstar", "lab"),
             ("prepared_fi_external", "multimodal", "lab"),
             ("prepared_fi_external", "stanford", "lab")]
FLASH_WS = (0, 1, 2)            # flash wait states examined for the worst case without cache


def page(n):
    return max(0x1000, (n + 0xFFF) & ~0xFFF)


def symbols():
    s = {}
    for line in open(os.path.join(HERE, "symbols.txt")):
        p = line.split()
        if len(p) == 3:
            s[p[2]] = int(p[0], 16)
    return s


def new_cpu(mode_name):
    if mode_name == "cortex-m4":
        mu = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
        try:
            mu.ctl_set_cpu_model(A.UC_CPU_ARM_CORTEX_M4)
        except Exception:
            pass
    else:
        mu = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
        mu.reg_write(A.UC_ARM_REG_C1_C0_2, mu.reg_read(A.UC_ARM_REG_C1_C0_2) | (0xF << 20))
        mu.reg_write(A.UC_ARM_REG_FPEXC, 0x40000000)
    mu.mem_map(FLASH, 0x40000)
    mu.mem_write(FLASH, open(os.path.join(HERE, "flash.bin"), "rb").read())
    mu.mem_write(STOP, b"\x00\xbf\x00\xbf")
    mu.mem_map(RAM, 0x10000)
    mu.mem_map(PAR, 0x1000)
    return mu


def emulate(mode_name, entry, x, params, hook=None):
    """x: float32 array (n,3). Returns (fi_logP (h,2), cue (h,), mu)."""
    mu = new_cpu(mode_name)
    xb = np.ascontiguousarray(x, np.float32).tobytes()
    n = len(x)
    mu.mem_map(IN, page(len(xb)))
    mu.mem_write(IN, xb)
    mu.mem_map(OUT, page(8 * (n // 16 + 2)))
    mu.mem_map(CUE, page(n // 16 + 2))
    mu.mem_write(PAR, params)
    mu.reg_write(A.UC_ARM_REG_SP, STACK_TOP)
    mu.reg_write(A.UC_ARM_REG_LR, STOP | 1)
    for r, v in ((A.UC_ARM_REG_R0, IN), (A.UC_ARM_REG_R1, n), (A.UC_ARM_REG_R2, OUT), (A.UC_ARM_REG_R3, CUE)):
        mu.reg_write(r, v)
    mu.mem_write(STACK_TOP - 8, PAR.to_bytes(4, "little"))        # 5th argument on the stack
    mu.reg_write(A.UC_ARM_REG_SP, STACK_TOP - 8)
    hs = []
    if hook:
        hs.append(mu.hook_add(UC_HOOK_CODE, hook[0]))
        if len(hook) > 1:
            hs.append(mu.hook_add(UC_HOOK_MEM_READ, hook[1], begin=FLASH, end=FLASH + 0x3FFFF))
    mu.emu_start(entry | 1, STOP)
    for h in hs:
        mu.hook_del(h)
    nh = mu.reg_read(A.UC_ARM_REG_R0)
    out = np.frombuffer(bytes(mu.mem_read(OUT, 8 * nh)), np.float32).reshape(-1, 2)
    cue = np.frombuffer(bytes(mu.mem_read(CUE, nh)), np.uint8).astype(bool)
    return out, cue


def cycle_fn():
    try:
        from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_MCLASS
        md = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS)
    except Exception:
        return None
    w = {"vldr": 2, "vstr": 2, "vmla": 3, "vmls": 3, "vfma": 3, "vfms": 3, "vnmla": 3, "vnmls": 3,
         "vdiv": 14, "vsqrt": 14, "ldr": 2, "ldrb": 2, "ldrh": 2, "sdiv": 6, "udiv": 6}
    def cyc(code, addr):
        for ins in md.disasm(code, addr, 1):
            m = ins.mnemonic.split(".")[0]
            if m in ("push", "pop", "vpush", "vpop", "ldm", "stm", "vldmia", "vstmia", "ldmia", "stmia"):
                return 1 + max(1, ins.op_str.count(",") + 1)
            if m.startswith("b") and m not in ("bic", "bics", "bfi", "bfc"):
                return 2
            return w.get(m, 1)
        return 1
    return cyc


def count(mode_name, entry, x, params):
    """instructions, TRM cycles, 32-bit instruction fetch words, flash data reads, per-address counts"""
    cyc = cycle_fn()
    cache, tot, per = {}, {"ins": 0, "cyc": 0, "fetch": 0.0, "fread": 0}, {}
    def on_code(uc, addr, size, _):
        tot["ins"] += 1
        tot["fetch"] += size / 4.0
        per[addr] = per.get(addr, 0) + 1
        if cyc is not None:
            c = cache.get(addr)
            if c is None:
                c = cache[addr] = cyc(bytes(uc.mem_read(addr, size)), addr)
            tot["cyc"] += c
    def on_read(uc, access, addr, size, value, _):
        tot["fread"] += 1
    emulate(mode_name, entry, x, params, (on_code, on_read))
    return tot, per


def hot_footprint(per_many, per_one, frac):
    """smallest code footprint (bytes) holding `frac` of the executed instructions per decision"""
    d = {a: per_many.get(a, 0) - per_one.get(a, 0) for a in per_many}
    items = sorted(((v, a) for a, v in d.items() if v > 0), reverse=True)
    total = float(sum(v for v, _ in items))
    acc, addrs = 0.0, []
    for v, a in items:
        acc += v
        addrs.append(a)
        if acc >= frac * total:
            break
    return len(addrs), (max(addrs) - min(addrs) + 4) if addrs else 0


def agreement(mode_used, S, frozen, role, ds, ctx, folder, recs_df, say_err):
    rows = []
    for r in recs_df.itertuples():
        acc = np.load(os.path.join(folder, r.rec + ".npz"))["acc"].astype(np.float64)
        if len(acc) < 300:
            continue
        for det, arr in DETS:
            cfg = frozen[ctx][det][str(int(C.MAIN_BUDGET))]
            ref, F, _ = R.ref_cue(K, acc, cfg, arr)
            out, cue = emulate(mode_used, S["run"], acc.astype(np.float32), R.params_bytes(cfg, arr))
            m = min(len(cue), len(ref))
            rows.append(dict(role=role, dataset=ds, context=ctx, rec=r.rec, detector=det, decisions=m,
                             agree=int((cue[:m] == ref[:m]).sum()), cue_on=int(ref[:m].sum()),
                             max_dfi=float(np.abs(out[:m, 0] - F[:m, 0]).max())))
        print("  %s %s done" % (ds, r.rec), flush=True)
    return rows


def main():
    S = symbols()
    frozen = json.load(open(os.path.join(C.OUT, "frozen", "frozen_fi.json")))
    lines = []
    def say(s):
        print(s, flush=True)
        lines.append(s)

    mode_used = None
    for mode_name in ("cortex-m4", "thumb2-vfp-fallback"):
        try:
            raw0 = open(os.path.join(HERE, "acc.bin"), "rb").read()
            n0 = int(np.frombuffer(raw0[:8], np.int64)[0])
            x0 = np.frombuffer(raw0[8:8 + 12 * n0], np.float32).reshape(n0, 3)
            emulate(mode_name, S["run"], x0[:16 * 40], R.params_bytes(frozen["lab"]["D4_D3_plus_arrest"]["10"], True))
            mode_used = mode_name
            break
        except UcError as e:
            say("emulator mode %s failed: %s" % (mode_name, e))
    if mode_used is None:
        say("RESULT: emulation failed")
        sys.exit(1)
    rows = []
    recs = pd.read_csv(os.path.join(C.PREP, "recordings.csv"))
    rng = np.random.RandomState(0)
    for ds in C.DEV_DATASETS:
        sub = recs[recs["dataset"] == ds]
        rows += agreement(mode_used, S, frozen, "development (10 random recordings)", ds, CONTEXT[ds], C.PREP,
                          sub.iloc[rng.permutation(len(sub))[:N_PER_DATASET]], say)
    for folder, ds, ctx in TEST_SETS:
        rp = os.path.join(folder, "recordings.csv")
        if not os.path.exists(rp):
            say("missing %s" % rp)
            continue
        t = pd.read_csv(rp)
        rows += agreement(mode_used, S, frozen, "test (all recordings)", ds, ctx, folder, t[t["dataset"] == ds], say)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "mcu_agreement.csv"), index=False)
    say("emulated core: %s" % mode_used)
    for (role, ds, det), g in df.groupby(["role", "dataset", "detector"], sort=False):
        say("%-22s %-20s %-24s recs %3d decisions %7d cue-on %6d agreement %.4f%% max|dFI| %.1e" % (
            role[:22], ds, det, len(g), g.decisions.sum(), g.cue_on.sum(), 100 * g.agree.sum() / g.decisions.sum(), g.max_dfi.max()))
    for role, g in df.groupby("role", sort=False):
        say("%s: cue-decision agreement %.4f%% of %d decisions in %d recordings (%d mismatches)" % (
            role, 100 * g.agree.sum() / g.decisions.sum(), g.decisions.sum(), g.rec.nunique(), g.decisions.sum() - g.agree.sum()))

    raw = open(os.path.join(HERE, "acc.bin"), "rb").read()
    n = int(np.frombuffer(raw[:8], np.int64)[0])
    x = np.frombuffer(raw[8:8 + 12 * n], np.float32).reshape(n, 3)
    cfg = frozen["lab"]["D4_D3_plus_arrest"][str(int(C.MAIN_BUDGET))]
    p = R.params_bytes(cfg, True)
    one, per1 = count(mode_used, S["run"], x[:16], p)
    many, perm = count(mode_used, S["run"], x[:16 * 61], p)
    k = 60.0
    ins = (many["ins"] - one["ins"]) / k
    cyc = (many["cyc"] - one["cyc"]) / k
    fetch = (many["fetch"] - one["fetch"]) / k
    fread = (many["fread"] - one["fread"]) / k
    say("full controller (D4, worst case), per decision:")
    say("  executed instructions %.0f, 32-bit instruction fetches %.0f, data reads from flash %.0f" % (ins, fetch, fread))
    for frac in (0.95, 0.99):
        na, span = hot_footprint(perm, per1, frac)
        say("  %.0f%% of executed instructions come from %d instructions in a %d-byte code region" % (100 * frac, na, span))
    if cyc > 0:
        say("  approx. core cycles (Cortex-M4 TRM timings): %.0f" % cyc)
        for ws in FLASH_WS:
            c = cyc + ws * (fetch + fread)
            ms = c / CLOCK_HZ * 1e3
            say("  flash wait states %d, no cache (upper bound): %.0f cycles = %.2f ms at 64 MHz, CPU load %.2f%% at %d decisions/s"
                % (ws, c, ms, ms * DECISIONS_PER_S / 10, DECISIONS_PER_S))
    say("  lower bound (1 cycle per instruction): %.2f ms at 64 MHz" % (ins / CLOCK_HZ * 1e3))
    for l in open(os.path.join(HERE, "size.txt")):
        if any(k in l for k in (".text", ".rodata", ".bss", ".data")):
            say("  " + l.strip())
    open(os.path.join(HERE, "mcu_results.txt"), "w").write("\n".join(lines) + "\n")
    print("saved mcu_results.txt and mcu_agreement.csv")


if __name__ == "__main__":
    main()
