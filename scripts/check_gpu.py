import time
import torch

def sync():
    torch.cuda.synchronize()

def main():
    print(f"torch: {torch.__version__}")
    print(f"HIP (ROCm): {torch.version.hip}")
    print(f"is GPU available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("GPU is not found")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}, VRAM: {props.total_memory / 1024**3:.1f} ГБ")
    dev = torch.device("cuda")

    # Does CPU and GPU results match
    a = torch.randn(256, 256)
    ok = torch.allclose(a @ a, (a.to(dev) @ a.to(dev)).cpu(), atol=1e-3)
    print(f"Does results of CPU and GPU match: {ok}")

    # Speed of matrix multiplication
    x = torch.randn(4096, 4096, device=dev)
    _ = x @ x
    sync()
    t = time.perf_counter()
    for _ in range(20):
        _ = x @ x
    sync()
    print(f"matmul 4096x4096 x20: {time.perf_counter() - t:.2f} с")

    try:
        import timm
    except ImportError:
        print("timm is not installed")
        return

    models = {
        "convnext_tiny": {},
        "convnext_base": {},
        "convnext_tiny.in12k_ft_in1k_384": {},
        "convnextv2_tiny": {},
        "convnextv2_base": {},
        "convnextv2_tiny.fcmae_ft_in22k_in1k_384": {},
        "tf_efficientnetv2_s": {},
        "vit_small_patch14_dinov2.lvd142m": {"img_size": 224},
        "eva02_small_patch14_336.mim_in22k_ft_in1k": {"img_size": 336},
        "swinv2_tiny_window16_256.ms_in1k": {"img_size": 256},
    }
    for name, kwargs in models.items():
        try:
            model = timm.create_model(name, pretrained=False, num_classes=0, **kwargs).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            if name == "eva02_small_patch14_336.mim_in22k_ft_in1k":
                imgs = torch.randn(32, 3, 336, 336, device=dev)
            elif name == "swinv2_tiny_window16_256.ms_in1k":
                imgs = torch.randn(32, 3, 256, 256, device=dev)
            else:
                imgs = torch.randn(32, 3, 224, 224, device=dev)
            for amp in (False, True):
                torch.cuda.reset_peak_memory_stats()
                scaler = torch.amp.GradScaler("cuda", enabled=amp)

                def step():
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        loss = model(imgs).float().pow(2).mean()
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    return loss

                step()
                sync()
                t = time.perf_counter()
                for _ in range(5):
                    loss = step()
                sync()
                dt = (time.perf_counter() - t) / 5
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"{name:36s} amp={amp!s:5s} {dt*1000:7.0f} ms/iter  "
                      f"memory peak {mem:.1f} GB  loss is finite: {bool(torch.isfinite(loss))}")
        except Exception as e:
            print(f"{name}: error {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()