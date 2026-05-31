"""
smoke_test.py
=============
Fast wiring check for the v2 pipeline. Run this BEFORE launching full training:

    cd src && python smoke_test.py

It uses tiny dummy tensors, a tiny temporary parquet, and one real FITS pair,
so it finishes in seconds and trains nothing. The model checks run on the SAME
device training will use (CUDA / MPS / CPU), so device-specific operator gaps
are caught here rather than mid-training. If everything passes you are clear to
run pretrain_ssl.py then train_real_pairs.py.
"""

import os
import sys
import tempfile
import traceback

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
_PASS, _FAIL = [], []


def check(name, fn):
    try:
        fn()
        _PASS.append(name)
        print(f"PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        _FAIL.append(name)
        print(f"FAIL  {name}  ->  {exc!r}")
        traceback.print_exc()


# ----------------------------------------------------------------------
def t_imports():
    import preprocessing_oiii  # noqa: F401
    import architectures_v2  # noqa: F401
    import datasets_v2  # noqa: F401


def t_preprocessing():
    import preprocessing_oiii as P
    w = P.MASTER_GRID
    L = len(w)
    g = lambda c, a, f: a * np.exp(-0.5 * ((w - c) / (f / 2.3548)) ** 2)
    flat = (g(5006.8, 45, 8) + g(4861, 20, 130) + g(6563, 25, 160)
            + np.random.default_rng(0).normal(0, 1, L))
    mad, _ = P.mad_normalize(flat)
    x, info = P.build_two_channel(mad, channel1_scale=0.02)
    assert x.shape == (2, L), x.shape
    assert info["oiii_reliable"] in (True, False)
    ch, meta = P.make_synthetic_change(mad, np.random.default_rng(1))
    assert ch.shape == (L,)
    assert "changed" in meta


def t_encoder():
    from architectures_v2 import SpectraEncoder, SEQ_LEN
    enc = SpectraEncoder(in_channels=2).to(DEVICE)
    x = torch.randn(4, 2, SEQ_LEN, device=DEVICE)
    assert tuple(enc.feature_map(x).shape) == (4, 256, 512)
    assert tuple(enc.embed(x).shape) == (4, 512)
    assert tuple(enc(x).shape) == (4, 512)


def t_masked_autoencoder():
    from architectures_v2 import (MaskedSpectraAutoencoder, apply_span_mask,
                                  SEQ_LEN)
    mae = MaskedSpectraAutoencoder(in_channels=2).to(DEVICE)
    x = torch.randn(4, 2, SEQ_LEN, device=DEVICE)
    xm, mask = apply_span_mask(x, mask_ratio=0.5)
    assert tuple(xm.shape) == (4, 2, SEQ_LEN)
    assert tuple(mask.shape) == (4, SEQ_LEN) and mask.dtype == torch.bool
    assert 0.2 < mask.float().mean().item() < 0.95
    recon = mae(xm)
    assert tuple(recon.shape) == (4, 2, SEQ_LEN), recon.shape


def t_ssl_train_step():
    from architectures_v2 import (MaskedSpectraAutoencoder, apply_span_mask,
                                  SEQ_LEN)
    from pretrain_ssl import masked_mse
    mae = MaskedSpectraAutoencoder(in_channels=2).to(DEVICE)
    opt = torch.optim.AdamW(mae.parameters(), lr=1e-4)
    x = torch.randn(4, 2, SEQ_LEN, device=DEVICE)
    valid = torch.ones(4, SEQ_LEN, dtype=torch.bool, device=DEVICE)
    xm, span = apply_span_mask(x, valid, mask_ratio=0.5)
    loss = masked_mse(mae(xm), x, span, valid)
    assert torch.isfinite(loss), loss
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in mae.encoder.parameters() if p.grad is not None]
    assert len(grads) > 0, "encoder received no gradient"
    opt.step()


def t_siamese():
    from architectures_v2 import SiameseChangeNet, SEQ_LEN
    net = SiameseChangeNet(in_channels=2).to(DEVICE)
    net.eval()  # disable dropout: order-invariance is a property of the
                # deterministic computation, not of the stochastic dropout mask
    x1 = torch.randn(4, 2, SEQ_LEN, device=DEVICE)
    x2 = torch.randn(4, 2, SEQ_LEN, device=DEVICE)
    with torch.no_grad():
        out = net(x1, x2)
        assert tuple(out.shape) == (4, 1), out.shape
        out_swapped = net(x2, x1)          # epoch-order invariance
        assert torch.allclose(out, out_swapped, atol=1e-5), "fusion not symmetric"


def t_siamese_train_step():
    from architectures_v2 import (SiameseChangeNet, BinaryFocalLossWithLogits,
                                  SEQ_LEN)
    net = SiameseChangeNet(in_channels=2).to(DEVICE)
    crit = BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    x1 = torch.randn(8, 2, SEQ_LEN, device=DEVICE)
    x2 = torch.randn(8, 2, SEQ_LEN, device=DEVICE)
    y = torch.randint(0, 2, (8, 1)).float().to(DEVICE)
    loss = crit(net(x1, x2), y)
    assert torch.isfinite(loss), loss
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in net.encoder.parameters() if p.grad is not None]
    assert len(grads) > 0, "encoder received no gradient through the siamese"
    opt.step()


def t_encoder_weight_transfer():
    from architectures_v2 import (MaskedSpectraAutoencoder, SiameseChangeNet,
                                  load_encoder_into)
    mae = MaskedSpectraAutoencoder(in_channels=2)
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "ssl_encoder.pth")
        torch.save({"encoder_state_dict": mae.encoder.state_dict()}, ckpt)
        net = SiameseChangeNet(in_channels=2)
        load_encoder_into(net, ckpt, device="cpu", verbose=False)
    a = dict(mae.encoder.named_parameters())
    b = dict(net.encoder.named_parameters())
    key = next(iter(a))
    assert torch.allclose(a[key], b[key]), "encoder weights did not transfer"


def t_ssl_dataset_tiny_parquet():
    import pandas as pd
    from preprocessing_oiii import MASTER_GRID
    from datasets_v2 import SSLSpectraDataset
    rng = np.random.default_rng(0)
    L = len(MASTER_GRID)
    cols = [str(w) for w in MASTER_GRID]
    df = pd.DataFrame(rng.normal(0, 1, (24, L)).astype(np.float32), columns=cols)
    df["agn_type"] = 1
    df["z"] = 0.1
    with tempfile.TemporaryDirectory() as td:
        pq = os.path.join(td, "tiny.parquet")
        df.to_parquet(pq)
        ds = SSLSpectraDataset([pq], channel1_scale=None, verbose=False)
        assert len(ds) == 24, len(ds)
        x0, v0 = ds[0]
        assert tuple(x0.shape) == (2, L), x0.shape
        assert tuple(v0.shape) == (L,) and v0.dtype == torch.bool


def t_real_fits_pair():
    """End-to-end: one real crossmatch pair -> 2-channel tensors."""
    import pandas as pd
    from datasets_v2 import fits_to_flat, _two_channel
    from preprocessing_oiii import mad_normalize, measure_oiii_flux, MASTER_GRID

    L = len(MASTER_GRID)
    pkl = os.path.join(BASE_DIR, "data/dr16_sdssv_crossmatch_lowz.pkl")
    spectra_dir = os.path.join(BASE_DIR, "data/dr16_sdssv_crossmatch")
    if not os.path.exists(pkl):
        raise FileNotFoundError("crossmatch pickle not found -- skipping")
    row = pd.read_pickle(pkl).iloc[0]
    z = float(row["z"])
    for col in ["specname_dr16", "specname_sdssv"]:
        path = os.path.join(spectra_dir, str(row[col]))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        flat, valid = fits_to_flat(path, z)
        assert flat.shape == (L,), flat.shape
        assert valid.shape == (L,) and valid.dtype == bool
        mad, _ = mad_normalize(flat, valid=valid)
        o, s = measure_oiii_flux(mad, valid=valid)
        x = _two_channel(mad, o, s >= 4.0 and o > 1e-6, 0.02)
        assert x.shape == (2, L)
        assert np.isfinite(x).all(), "non-finite values in 2-channel tensor"


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"v2 pipeline smoke test  (device: {DEVICE})")
    print("=" * 60)
    check("imports", t_imports)
    check("preprocessing_oiii", t_preprocessing)
    check("SpectraEncoder forward/shapes", t_encoder)
    check("MaskedSpectraAutoencoder + span mask", t_masked_autoencoder)
    check("SSL training step (loss + backward)", t_ssl_train_step)
    check("SiameseChangeNet forward + order-invariance", t_siamese)
    check("Siamese training step (focal loss + backward)", t_siamese_train_step)
    check("SSL encoder -> Siamese weight transfer", t_encoder_weight_transfer)
    check("SSLSpectraDataset on a tiny parquet", t_ssl_dataset_tiny_parquet)
    check("real FITS pair -> 2-channel tensors", t_real_fits_pair)

    print("=" * 60)
    print(f"PASSED {len(_PASS)} / {len(_PASS) + len(_FAIL)}")
    if _FAIL:
        print("FAILED: " + ", ".join(_FAIL))
        sys.exit(1)
    print("All checks passed -- you are clear to run:")
    print("  1) python pretrain_ssl.py      (Stage 1)")
    print("  2) python train_real_pairs.py  (Stage 2)")
    sys.exit(0)
