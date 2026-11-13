#!/usr/bin/env python3
"""F — viabilidade on-device: tamanho, latência em CPU (lote 1) e conversão Core ML.

Uso: python3 on_device.py
Mede backbone + cabeça linear (ridge exportada como camada Linear) para os modelos testados e
tenta converter o melhor com coremltools (mlprogram, FP16). No Linux o coremltools converte e
salva o .mlpackage, mas não executa (a execução exige macOS/iOS).
Grava <TD>/resultados/on_device.json e <TD>/pesos/*.mlpackage (fora do repositório).
"""
import json, os, time
import numpy as np, torch, timm
from comum import TD

PESOS = os.path.join(TD, "pesos")
CFG = {
    "vits16_in21k": ("vit_small_patch16_224", (224, 288)),
    "siglip_b16": ("vit_base_patch16_siglip_224", (224, 224)),
    "effb0_ft": ("efficientnet_b0", (240, 320)),
}


class Cabeca(torch.nn.Module):
    """Backbone congelado + média dos patches/token agregado + camada linear (4 saídas em log)."""
    def __init__(self, m, dim_saida=4):
        super().__init__()
        self.m = m
        d = m.num_features * 2
        self.lin = torch.nn.Linear(d, dim_saida)

    def forward(self, x):
        t = self.m.forward_features(x)
        pool = self.m.forward_head(t, pre_logits=True)
        return self.lin(torch.cat([pool, t[:, self.m.num_prefix_tokens:].mean(1)], 1))


def constroi(nome):
    arch, hw = CFG[nome]
    if nome == "effb0_ft":
        m = timm.create_model(arch, pretrained=False, num_classes=4)
        p = os.path.join(PESOS, "effb0_ft.pt")
        if os.path.exists(p):
            m.load_state_dict(torch.load(p))
        return m.eval(), hw
    m = timm.create_model(arch, pretrained=False, num_classes=0, dynamic_img_size=False, img_size=hw)
    return Cabeca(m).eval(), hw


def latencia(m, hw, threads, n=10):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, *hw)
    with torch.inference_mode():
        for _ in range(2):
            m(x)
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); m(x); t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def main():
    res = {}
    for nome in CFG:
        m, hw = constroi(nome)
        npar = sum(p.numel() for p in m.parameters())
        res[nome] = dict(parametros_M=npar / 1e6, fp16_MB=npar * 2 / 2**20, entrada=hw,
                         lat_ms_1thread=latencia(m, hw, 1), lat_ms_4threads=latencia(m, hw, 4))
        print(nome, res[nome], flush=True)
    import coremltools as ct
    for nome in ("effb0_ft", "vits16_in21k"):
        m, hw = constroi(nome)
        try:
            tr = torch.jit.trace(m, torch.randn(1, 3, *hw))
            ml = ct.convert(tr, inputs=[ct.TensorType(name="imagem", shape=(1, 3, *hw))],
                            convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                            minimum_deployment_target=ct.target.iOS16)
            out = os.path.join(PESOS, f"{nome}.mlpackage")
            ml.save(out)
            tam = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
            res[nome]["coreml"] = f"ok ({tam / 2**20:.1f} MB, mlprogram FP16, iOS16+)"
        except Exception as e:  # noqa
            res[nome]["coreml"] = f"falhou: {type(e).__name__}: {str(e)[:200]}"
        print(nome, res[nome]["coreml"], flush=True)
    with open(os.path.join(TD, "resultados", "on_device.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
