#!/usr/bin/env python3
"""Seção 8 — ponto de entrada único: treina o modelo final escolhido (uma foto de cima -> massa,
proteína, kcal, macros e massa por grupo) e exporta para Core ML.

Uso: OMP_NUM_THREADS=4 python3 zen_final.py [--membros ft,mlp,rl] [--sem-coreml]
O modelo escolhido pela CV entre os que usam um único backbone (SigLIP 2 B/16, 224 px) é a média de
três cabeças sobre o MESMO backbone:
  ft   topo ajustado (3 últimos blocos + agregação MAP) + camada linear   -> zen_finetune.py final
  mlp  MLP multitarefa sobre as features congeladas [agregado ‖ média]
  rl   ridge sobre as features congeladas, treinado com 8 orientações + quadros laterais
  (r8  ridge sobre as features congeladas, 8 orientações; opcional)
Os 9 primeiros blocos são compartilhados; o custo extra das três cabeças é um segundo topo de 3 blocos.

O que este script faz, na ordem (cada passo é pulado se a saída já existe em <TD>):
  1. zen_dados.py -> lista de treino estrita e alvos;  2. zen_extrai.py siglip2_b16 --orient 8 --tokens 9;
  3. zen_finetune.py final (se 'ft' está nos membros);  4. treina as cabeças mlp/ridge;
  5. monta o modelo completo imagem -> saídas em log e confere com o banco de previsões em pratos do teste;
  6. mede tamanho e latência em CPU;  7. exporta Core ML (mlprogram FP16 e pesos em 8 bits).
Grava <TD>/pesos/zen_final.pt, zen_final_*.mlpackage e <TD>/resultados/zen_final.json (fora do repositório).

Saída do modelo: tensor [1, membros, 22] em log, uma orientação por chamada. Como foi avaliado: média das
4 orientações que preservam o quadro para 'ft' e das 8 orientações D4 para as cabeças congeladas; depois,
média entre os membros. Ordem das 22 saídas: log(massa g), log1p(proteína g), log1p(kcal), log(densidade+0,01),
log1p(gordura g), log1p(carboidrato g), log1p(massa por grupo ×8), log1p(proteína por grupo ×8);
grupos = eval/grupos_alimentos.GRUPOS. Entrada: foto zenital inteira redimensionada (sem recorte) para
224×224, RGB, (x/255 − 0,5)/0,5.
"""
import copy, json, os, subprocess, sys, time, warnings
import numpy as np, torch
from PIL import Image
from zen_comum import TD, Dados, Feat, ridge_cv, ridge_final, N_ALVOS, GRUPOS
from zen_extrai import orienta

PESOS = os.path.join(TD, "pesos")
AQUI = os.path.dirname(os.path.abspath(__file__))
BL = ("agregado", "media")
N_OR = {"ft": 4, "mlp": 8, "rl": 8, "r8": 8}
BANCO = {"ft": "ft cv", "mlp": "mlp siglip2_b16 o8", "rl": "ridge siglip2_b16 o8 + quadros laterais no treino", "r8": "ridge siglip2_b16 o8"}


def garante(saida, *cmd):
    if not os.path.exists(saida):
        print("rodando:", " ".join(cmd), flush=True)
        subprocess.run([sys.executable, os.path.join(AQUI, cmd[0]), *cmd[1:]], check=True)


def linear(W, b):
    """Camada linear a partir de W [entradas, saídas] e b [saídas] (numpy)."""
    lin = torch.nn.Linear(W.shape[0], W.shape[1])
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(W.T.astype(np.float32))); lin.bias.copy_(torch.from_numpy(b.astype(np.float32)))
    return lin


def de_ridge(r):
    return linear(r.W / r.sd[:, None], r.mt - (r.mu / r.sd) @ r.W)


class ModeloZenital(torch.nn.Module):
    """SigLIP 2 B/16: 9 blocos compartilhados -> topo congelado (cabeças mlp/ridge) e topo ajustado (cabeça ft)."""

    def __init__(self, vit, topo_ft, cabecas_cong, cabeca_ft):
        super().__init__()
        self.vit, self.topo_ft, self.cabeca_ft = vit, topo_ft, cabeca_ft
        self.cabecas_cong = torch.nn.ModuleList(cabecas_cong)

    def forward(self, x):
        v = self.vit
        t = v.norm_pre(v.patch_drop(v._pos_embed(v.patch_embed(x))))
        for b in v.blocks[:9]:
            t = b(t)
        saidas = []
        if self.cabeca_ft is not None:
            tb = self.topo_ft["norm"](self.topo_ft["blocks"](t))
            fb = torch.cat([self.topo_ft["fc_norm"](self.topo_ft["attn_pool"](tb)), tb.mean(1)], 1)
            saidas.append(self.cabeca_ft(fb))
        if len(self.cabecas_cong):
            ta = t
            for b in v.blocks[9:]:
                ta = b(ta)
            ta = v.norm(ta)
            fa = torch.cat([v.forward_head(ta, pre_logits=True), ta.mean(1)], 1)
            saidas += [c(fa) for c in self.cabecas_cong]
        return torch.stack(saidas, 1)


def constroi(membros, D):
    import timm
    from timm.models.vision_transformer import _load_weights
    vit = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0, dynamic_img_size=False)
    _load_weights(vit, os.path.join(PESOS, "siglip2_b16_224.npz"))
    topo_ft, cab_ft, cong = None, None, []
    for m in membros:   # a ordem das saídas é: ft primeiro, depois as cabeças congeladas na ordem pedida
        if m == "ft":
            ck = torch.load(os.path.join(PESOS, "zen_ft.pt"), weights_only=False)
            e = ck["estado"]
            pega = lambda pre: {k[len(pre):]: v for k, v in e.items() if k.startswith(pre)}
            blocks = copy.deepcopy(vit.blocks[9:]); blocks.load_state_dict({k: v for k, v in pega("blocks.").items()})
            norm = copy.deepcopy(vit.norm); norm.load_state_dict(pega("norm."))
            ap = copy.deepcopy(vit.attn_pool); ap.load_state_dict(pega("attn_pool."))
            fn = copy.deepcopy(vit.fc_norm); fn.load_state_dict(pega("fc_norm."))
            topo_ft = torch.nn.ModuleDict(dict(blocks=blocks, norm=norm, attn_pool=ap, fc_norm=fn))
            ts, tm = ck["ts"], ck["tm"]   # desfaz a padronização dos alvos dentro da camada linear
            cab_ft = linear((e["head.weight"][:N_ALVOS].numpy() * ts[:, None]).T, e["head.bias"][:N_ALVOS].numpy() * ts + tm)
        elif m == "r8":
            X = Feat("siglip2_b16").pega(D.ids, tuple(range(8)), BL)
            _, alfa, _ = ridge_cv(X, D.T, D.fold)
            cong.append(de_ridge(ridge_final(X, D.T, alfa)[1]))
        elif m == "rl":
            from zen_experimentos import dados_laterais, ridge_lateral
            cong.append(de_ridge(ridge_lateral(D, Feat("siglip2_b16"), *dados_laterais(D))[3]))
        elif m == "mlp":
            from zen_experimentos import treina_mlp
            X = Feat("siglip2_b16").pega(D.ids, tuple(range(8)), BL)
            _, k = treina_mlp(X, D.T, [], devolve_modelo=True)
            l1, l2 = k["net"][1], k["net"][4]   # Dropout, Linear, GELU, Dropout, Linear
            W1 = l1.weight.detach().numpy().T / k["sd"][:, None]; b1 = l1.bias.detach().numpy() - (k["mu"] / k["sd"]) @ l1.weight.detach().numpy().T
            W2 = l2.weight.detach().numpy()[:N_ALVOS].T * k["ts"]; b2 = l2.bias.detach().numpy()[:N_ALVOS] * k["ts"] + k["tm"]
            cong.append(torch.nn.Sequential(linear(W1, b1), torch.nn.GELU(), linear(W2, b2)))
    return ModeloZenital(vit, topo_ft, cong, cab_ft).eval()


def imagem(path):
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    return (torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1) - 0.5) / 0.5


def preve(mod, membros, x):
    """Como na avaliação: média das orientações de cada membro (4 para ft, 8 para os demais) e média entre membros."""
    ordem = [m for m in membros if m == "ft"] + [m for m in membros if m != "ft"]
    with torch.inference_mode():
        o = torch.stack([mod(orienta(x, k)) for k in range(8)])       # [8, B, membros, 22]
    por_membro = [o[:N_OR[m], :, i].mean(0) for i, m in enumerate(ordem)]
    return torch.stack(por_membro).mean(0).numpy(), {m: p.numpy() for m, p in zip(ordem, por_membro)}


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
    membros = arg("--membros", "ft,mlp,rl").split(",")
    garante(os.path.join(TD, "meta", "zen_treino.csv"), "zen_dados.py", TD)
    garante(os.path.join(TD, "feat", "zen_siglip2_b16_tokL9.npy"), "zen_extrai.py", TD, "siglip2_b16", "--orient", "8", "--tokens", "9")
    if "ft" in membros:
        garante(os.path.join(PESOS, "zen_ft.pt"), "zen_finetune.py", "final")
    D = Dados()
    t0 = time.time()
    mod = constroi(membros, D)
    res = dict(membros=membros, parametros_M=sum(p.numel() for p in mod.parameters()) / 1e6,
               saidas=["log_massa", "log1p_proteina", "log1p_kcal", "log_densidade", "log1p_gordura", "log1p_carbo"]
               + [f"log1p_massa_{g}" for g in GRUPOS] + [f"log1p_proteina_{g}" for g in GRUPOS])
    torch.save(mod.state_dict(), os.path.join(PESOS, "zen_final.pt"))
    # confere com o banco (previsões dos experimentos) em 16 pratos do teste
    te = D.aval["teste"]
    x = torch.stack([imagem(os.path.join(TD, "img", d, "rgb.png")) for d in te["chave"][:16]])
    out, por_membro = preve(mod, membros, x)
    res["dif_max_vs_banco_log"] = {}
    for m, p in por_membro.items():
        pb = os.path.join(TD, "resultados", "zen_banco", BANCO[m] + ".npz")
        if os.path.exists(pb):
            res["dif_max_vs_banco_log"][m] = float(np.abs(p - np.load(pb)["teste"][:16]).max())
    print("conferência com o banco, diferença máxima em log por membro:", {k: round(v, 4) for k, v in res["dif_max_vs_banco_log"].items()})
    print("exemplo (1º prato do teste):", {"massa_g": round(float(np.exp(out[0, 0])), 1), "proteina_g": round(float(np.expm1(out[0, 1])), 1),
                                            "real": (float(te["massa"][0]), float(te["proteina"][0]))})
    res["lat_ms_1thread"] = latencia(mod, 1); res["lat_ms_4threads"] = latencia(mod, 4)
    res["fp32_MB"] = os.path.getsize(os.path.join(PESOS, "zen_final.pt")) / 2 ** 20
    print(f"{res['parametros_M']:.1f} M parâmetros | latência em CPU x86 por orientação: {res['lat_ms_1thread']:.0f} ms (1 thread), {res['lat_ms_4threads']:.0f} ms (4 threads)")
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
