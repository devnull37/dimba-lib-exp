import torch
src = open("/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad/selfcorrect_test.py").read().split("PROMPTS = [")[0]
exec(src)
for name, path in [("repair-v1 (3k, random-token)", "/workspace/dimba-lib-exp/checkpoints/mdm_repair/mdm_sft_final.pt"),
                   ("repair-v2 @3k (self-corrupt)", "/workspace/dimba-lib-exp/checkpoints/mdm_repair2/mdm_sft_latest.pt")]:
    model, mid = load(path)
    fr, kr = detect_test(model, mid)
    print(f"{name}: fixes {fr*100:.1f}% | keeps clean {kr*100:.1f}%", flush=True)
    del model; torch.cuda.empty_cache()
print("QUICK_DETECT_DONE", flush=True)
