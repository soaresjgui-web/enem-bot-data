#!/usr/bin/env python3
"""Seção 8 — ponto de entrada único: treina o modelo final escolhido (uma foto de cima -> massa,
proteína, kcal, macros e massa por grupo) e exporta para Core ML.

Uso: OMP_NUM_THREADS=4 python3 zen_final.py [--cabeca ft|ridge] [--blocos agregado,media] [--sem-coreml]
Pré-requisitos (ver "Reproduzir" em docs/dados/precisao/resultado_modelo_treinado.md):
  zen_dados.py -> zen_extrai.py siglip2_b16 --orient 8 --tokens 9 [-> zen_finetune.py final]
Passos:
  1. cabeça final: 'ridge' = ridge multialvo sobre as features congeladas do SigLIP 2 B/16 com as
     8 orientações D4 como amostras (alfas pela CV agrupada); 'ft' = topo ajustado por zen_finetune.py final;
  2. monta o modelo completo imagem -> 22 saídas em log (uma orientação por chamada; o app faz a média
     das orientações) e confere com as previsões do banco em pratos do teste;
  3. mede tamanho e latência em CPU; 4. exporta Core ML (mlprogram FP16 e pesos em 8 bits).
Grava <TD>/pesos/zen_final.pt, zen_final*.mlpackage e <TD>/resultados/zen_final.json (fora do repositório).

Saídas (ordem): log(massa g), log1p(proteína g), log1p(kcal), log(densidade+0,01), log1p(gordura g),
log1p(carboidrato g), log1p(massa por grupo ×8), log1p(proteína por grupo ×8); grupos = eval/grupos_alimentos.GRUPOS.
Entrada: foto zenital inteira redimensionada (sem recorte) para 224×224, RGB, (x/255 − 0,5)/0,5.
"""
import json, os, sys, time, warnings
import numpy as np, torch
from PIL import Image
from zen_comum import TD, Dados, Feat, ridge_cv, ridge_final, N_ALVOS, GRUPOS
from zen_extrai import orienta

PESOS = os.path.join(TD, "pesos")


class ModeloZenital(torch.nn.Module):
    """SigLIP 2 B/16 + cabeça linear sobre os blocos de features escolhidos."""

    def __init__(self, vit, blocos, n_saidas=N_ALVOS):
        super().__init__()
        self.vit, self.blocos = vit, tuple(blocos)
        d = 768 * (len([b for b in blocos if b != "grade"]) + (4 if "grade" in blocos else 0))
        self.lin = torch.nn.Linear(d, n_saidas)

    def forward(self, x):
        t = self.vit.forward_features(x)
        f = []
        if "agregado" in self.blocos:
            f.append(self.vit.forward_head(t, pre_logits=True))
        if "media" in self.blocos:
            f.append(t.mean(1))
        if "gem" in self.blocos:
            f.append(t.clamp(min=0).pow(3).mean(1).pow(1 / 3))
        if "grade" in self.blocos:
            g = t.reshape(t.shape[0], 14, 14, t.shape[-1])
            f += [g[:, :7, :7].mean((1, 2)), g[:, :7, 7:].mean((1, 2)), g[:, 7:, :7].mean((1, 2)), g[:, 7:, 7:].mean((1, 2))]
        return self.lin(torch.cat(f, 1))


def vit_base():
    import timm
    from timm.models.vision_transformer import _load_weights
    m = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0, dynamic_img_size=False)
    _load_weights(m, os.path.join(PESOS, "siglip2_b16_224.npz"))
    return m


def constroi(cabeca, blocos, D):
    vit = vit_base()
    mod = ModeloZenital(vit, blocos)
    if cabeca == "ridge":
        F = Feat("siglip2_b16")
        X = F.pega(D.ids, tuple(range(8)), blocos)
        _, alfa, _ = ridge_cv(X, D.T, D.fold)
        _, r = ridge_final(X, D.T, alfa)
        with torch.no_grad():
            mod.lin.weight.copy_(torch.from_numpy((r.W / r.sd[:, None]).T.astype(np.float32)))
            mod.lin.bias.copy_(torch.from_numpy((r.mt - (r.mu / r.sd) @ r.W).astype(np.float32)))
        n_or = 8
    else:
        ck = torch.load(os.path.join(PESOS, "zen_ft.pt"), weights_only=False)
        e = ck["estado"]
        for i in range(3):
            vit.blocks[9 + i].load_state_dict({k[len(f"blocks.{i}."):]: v for k, v in e.items() if k.startswith(f"blocks.{i}.")})
        vit.norm.load_state_dict({k[5:]: v for k, v in e.items() if k.startswith("norm.")})
        vit.attn_pool.load_state_dict({k[10:]: v for k, v in e.items() if k.startswith("attn_pool.")})
        vit.fc_norm.load_state_dict({k[8:]: v for k, v in e.items() if k.startswith("fc_norm.")})
        ts, tm = torch.from_numpy(ck["ts"].astype(np.float32)), torch.from_numpy(ck["tm"].astype(np.float32))
        with torch.no_grad():   # desfaz a padronização dos alvos dentro da camada linear
            mod.lin.weight.copy_(e["head.weight"][:N_ALVOS] * ts[:, None])
            mod.lin.bias.copy_(e["head.bias"][:N_ALVOS] * ts + tm)
        n_or = 4
    return mod.eval(), n_or


def imagem(path):
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    return (torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1) - 0.5) / 0.5


def latencia(mod, threads, n=10):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        for _ in range(2):
            mod(x)
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); mod(x); t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def tamanho(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 2 ** 20


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    cabeca = arg("--cabeca", "ft")
    blocos = arg("--blocos", "agregado,media").split(",")
    D = Dados()
    t0 = time.time()
    mod, n_or = constroi(cabeca, blocos, D)
    res = dict(cabeca=cabeca, blocos=blocos, orientacoes=n_or, parametros_M=sum(p.numel() for p in mod.parameters()) / 1e6,
               saidas=["log_massa", "log1p_proteina", "log1p_kcal", "log_densidade", "log1p_gordura", "log1p_carbo"]
               + [f"log1p_massa_{g}" for g in GRUPOS] + [f"log1p_proteina_{g}" for g in GRUPOS])
    torch.save(mod.state_dict(), os.path.join(PESOS, "zen_final.pt"))
    # confere com o banco (previsões do experimento) em 8 pratos do teste
    nome_banco = {"ridge": "ridge siglip2_b16 o8" + ("+gem" if "gem" in blocos else "") + ("+grade" if "grade" in blocos else ""),
                  "ft": "ft cv"}[cabeca]
    pb = os.path.join(TD, "resultados", "zen_banco", nome_banco + ".npz")
    te = D.aval["teste"]
    with torch.inference_mode():
        x = torch.stack([imagem(os.path.join(TD, "img", d, "rgb.png")) for d in te["chave"][:8]])
        out = torch.stack([mod(orienta(x, k)) for k in range(n_or)]).mean(0).numpy()
    if os.path.exists(pb):
        ref = np.load(pb)["teste"][:8]
        res["dif_max_vs_banco_log"] = float(np.abs(out - ref).max())
        print(f"conferência com o banco ({nome_banco}): diferença máxima em log = {res['dif_max_vs_banco_log']:.4f}")
    print("exemplo (prato do teste):", {"massa_g": round(float(np.exp(out[0, 0])), 1), "proteina_g": round(float(np.expm1(out[0, 1])), 1),
                                         "real": (float(te["massa"][0]), float(te["proteina"][0]))})
    res["lat_ms_1thread"] = latencia(mod, 1); res["lat_ms_4threads"] = latencia(mod, 4)
    res["fp32_MB"] = os.path.getsize(os.path.join(PESOS, "zen_final.pt")) / 2 ** 20
    print(f"{res['parametros_M']:.1f} M parâmetros | latência CPU x86 por orientação: {res['lat_ms_1thread']:.0f} ms (1 thread), {res['lat_ms_4threads']:.0f} ms (4 threads)")
    if "--sem-coreml" not in sys.argv:
        warnings.filterwarnings("ignore")
        import coremltools as ct
        try:
            tr = torch.jit.trace(mod, torch.randn(1, 3, 224, 224), strict=False)
            ml = ct.convert(tr, inputs=[ct.ImageType(name="imagem", shape=(1, 3, 224, 224), scale=2 / 255.0, bias=[-1, -1, -1],
                                                     color_layout=ct.colorlayout.RGB)],
                            outputs=[ct.TensorType(name="saidas_log")], convert_to="mlprogram",
                            compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
            p16 = os.path.join(PESOS, "zen_final_fp16.mlpackage"); ml.save(p16)
            res["coreml_fp16_MB"] = tamanho(p16)
            print(f"Core ML FP16: {res['coreml_fp16_MB']:.1f} MB ({p16})")
            try:
                import coremltools.optimize.coreml as cto
                cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512))
                ml8 = cto.linear_quantize_weights(ml, config=cfg)
                p8 = os.path.join(PESOS, "zen_final_int8.mlpackage"); ml8.save(p8)
                res["coreml_int8_MB"] = tamanho(p8)
                print(f"Core ML com pesos em 8 bits: {res['coreml_int8_MB']:.1f} MB ({p8})")
            except Exception as e:  # noqa
                res["coreml_int8"] = f"falhou: {type(e).__name__}: {str(e)[:200]}"; print(res["coreml_int8"])
        except Exception as e:  # noqa
            res["coreml"] = f"falhou: {type(e).__name__}: {str(e)[:300]}"; print(res["coreml"])
    res["seg"] = time.time() - t0
    with open(os.path.join(TD, "resultados", "zen_final.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
