"""Runs the OFFICIAL Meta Audiobox Aesthetics code and the OFFICIAL SongEval code (MuQ-large-msd-iter features + the
SongEval scoring head) on synthetic test audio and on SongEval's own example.mp3.

It saves the end-to-end scores plus intermediate tensors, so the same models run on another machine can be checked
number for number against these reference outputs.

No copyrighted music is used: only generated test signals and the example file shipped in the SongEval repository."""
import json, os, sys, time, traceback
import numpy as np
import soundfile as sf

OUT = "refs"
os.makedirs(OUT, exist_ok=True)
SONGEVAL = os.path.abspath("SongEval")
STATUS = {}


def run(name, fn):
    t0 = time.time()
    try:
        fn()
        STATUS[name] = f"ok ({time.time() - t0:.1f}s)"
    except Exception:
        STATUS[name] = "FAILED:\n" + traceback.format_exc()
        print(f"[{name}] FAILED\n{traceback.format_exc()}", flush=True)
    json.dump(STATUS, open(f"{OUT}/status.json", "w"), indent=2)


def synth(sr, seconds=12.0, variant=0):
    """Deterministic music-like test signal: sustained chord + kick + snare + noise floor."""
    rng = np.random.default_rng(1234 + variant)
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t)
    chord = [130.81, 155.56, 196.0, 261.63] if variant == 0 else [116.54, 146.83, 174.61, 233.08]
    for k, f in enumerate(chord):
        y += 0.12 * np.sin(2 * np.pi * f * t + k) * (1 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    beat = 60 / 118
    for b in np.arange(0, seconds, beat):
        i = int(b * sr); n = min(int(0.25 * sr), len(t) - i)
        tt = np.arange(n) / sr
        y[i:i + n] += 0.5 * np.sin(2 * np.pi * (50 + 80 * np.exp(-tt / 0.03)) * tt) * np.exp(-tt / 0.12)
    for b in np.arange(beat, seconds, 2 * beat):
        i = int(b * sr); n = min(int(0.18 * sr), len(t) - i)
        y[i:i + n] += 0.2 * rng.standard_normal(n) * np.exp(-np.arange(n) / sr / 0.05)
    if variant:
        # some content near the 8 kHz / 12 kHz Nyquist limits, so the resamplers' filters are exercised
        y += 0.03 * np.sin(2 * np.pi * 9000 * t) + 0.02 * np.sin(2 * np.pi * 11500 * t)
    y += 0.01 * rng.standard_normal(len(t))
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def versions():
    import importlib.metadata as md
    v = {}
    for p in ["torch", "torchaudio", "transformers", "librosa", "numpy", "scipy", "muq", "audiobox_aesthetics", "soxr",
              "soundfile", "huggingface_hub", "hydra-core", "omegaconf", "safetensors", "einops"]:
        try:
            v[p] = md.version(p)
        except Exception as e:
            v[p] = f"? {e}"
    v["python"] = sys.version
    json.dump(v, open(f"{OUT}/versions.json", "w"), indent=2)
    print(v, flush=True)


def make_inputs():
    import librosa
    sf.write(f"{OUT}/synth_16k.wav", synth(16000), 16000, subtype="FLOAT")
    sf.write(f"{OUT}/synth_24k.wav", synth(24000), 24000, subtype="FLOAT")
    st = np.stack([synth(44100, variant=0), synth(44100, variant=1)], 1)
    sf.write(f"{OUT}/synth44_stereo.wav", st, 44100, subtype="FLOAT")
    for sr in (16000, 24000):
        w, _ = librosa.load(f"{SONGEVAL}/example/example.mp3", sr=sr)
        sf.write(f"{OUT}/example_{sr // 1000}k.wav", w, sr, subtype="FLOAT")


def audiobox():
    import torch
    torch.set_grad_enabled(False)
    from audiobox_aesthetics.infer import initialize_predictor, make_inference_batch
    pred = initialize_predictor()                      # official: facebook/audiobox-aesthetics from the Hub
    res = {}
    for nm, path in [("synth", f"{OUT}/synth_16k.wav"), ("example", f"{OUT}/example_16k.wav"),
                     ("synth44_stereo", f"{OUT}/synth44_stereo.wav"),
                     ("example_mp3", f"{SONGEVAL}/example/example.mp3")]:
        try:
            res[nm] = pred.forward([{"path": path}])[0]
        except Exception:
            res[nm] = {"error": traceback.format_exc()}
        print("audiobox", nm, res[nm], flush=True)
        json.dump(res, open(f"{OUT}/audiobox_reference.json", "w"), indent=2)
    # internals on synth_16k (12 s -> chunk 0 is full, chunk 1 is 2 s + zero padding)
    m = pred.model
    wavs = pred.audio_resample_mono([{"path": f"{OUT}/synth_16k.wav"}])
    wv, mk, wt, bd = make_inference_batch(wavs, 10, 10, sample_rate=16000)
    wav_b, mask_b = torch.stack(wv), torch.stack(mk)
    with torch.inference_mode():
        raw = m({"wav": wav_b, "mask": mask_b})
        (_, outs), epm = m.wavlm_model.extract_features(source=wav_b.squeeze(1), padding_mask=~mask_b.squeeze(1),
                                                        output_layer=13, ret_layer_results=True)
    d = {f"raw_{k}": v.float().numpy() for k, v in raw.items()}
    d["weights"] = np.array(wt, np.float32)
    for li in (0, 1, 6, 12):
        d[f"L{li}"] = outs[li][0].transpose(0, 1).float().numpy()       # (chunks, T, 768)
    d["frame_pad"] = epm.numpy()
    np.savez(f"{OUT}/audiobox_synth_internals.npz", **d)
    # what the official pipeline feeds the model for a 44.1 kHz stereo file (channel mean + torchaudio resample)
    w = pred.audio_resample_mono([{"path": f"{OUT}/synth44_stereo.wav"}])[0]
    np.save(f"{OUT}/synth44_stereo_to16k_official.npy", w.float().numpy())


def songeval():
    import torch, librosa
    torch.set_grad_enabled(False)
    sys.path.insert(0, SONGEVAL)
    from muq import MuQ
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from safetensors.torch import load_file
    # exactly as SongEval/eval.py
    cfg = OmegaConf.load(f"{SONGEVAL}/config.yaml")
    head = instantiate(cfg.generator).eval()
    missing = head.load_state_dict(load_file(f"{SONGEVAL}/ckpt/model.safetensors", device="cpu"), strict=False)
    print("songeval head load:", missing, flush=True)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").eval()
    names = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]
    res = {}
    for nm, path in [("synth", f"{OUT}/synth_24k.wav"), ("example", f"{OUT}/example_24k.wav"),
                     ("synth44_stereo", f"{OUT}/synth44_stereo.wav"),
                     ("example_mp3", f"{SONGEVAL}/example/example.mp3")]:
        wav, _ = librosa.load(path, sr=24000)
        if nm == "synth44_stereo":
            np.save(f"{OUT}/synth44_stereo_to24k_librosa.npy", wav)
        with torch.no_grad():
            out = muq(torch.tensor(wav).unsqueeze(0), output_hidden_states=True)
            hs = out["hidden_states"]
            s = head(hs[6]).squeeze(0)
        res[nm] = {k: float(v) for k, v in zip(names, s)}
        res[nm]["n_hidden_states"] = len(hs)
        print("songeval", nm, res[nm], flush=True)
        json.dump(res, open(f"{OUT}/songeval_reference.json", "w"), indent=2)
        if nm in ("synth", "example"):
            np.save(f"{OUT}/{nm}_muq_h6.npy", hs[6][0].float().numpy())
        if nm == "synth":
            for i in (0, 1, 12):
                np.save(f"{OUT}/synth_muq_h{i}.npy", hs[i][0].float().numpy())
            x = muq.model.preprocessing(torch.tensor(wav).unsqueeze(0), features=["melspec_2048"])
            x = muq.model.normalize(x)
            np.save(f"{OUT}/synth_muq_mel_norm.npy", x["melspec_2048"][0].float().numpy())


if __name__ == "__main__":
    run("versions", versions)
    run("inputs", make_inputs)
    run("audiobox", audiobox)
    run("songeval", songeval)
    print(json.dumps(STATUS, indent=2))
