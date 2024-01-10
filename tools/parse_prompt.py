import os
import re
import json

NUM_PROTO = int(os.environ.get("NUM_PROTO", 26))
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "")
# experiment/molecular_ebv_others_full_train
assert EXPERIMENT_NAME != ""

# def remove_class_name(class_name, ctx):
#     replacement = {   # replace class with super-class
#         "lung adenocarcinoma": "cancer",
#         "adenocarcinoma": "cancer",
#         "lung squamous cell carcinoma": "cancer",
#         "squamous cell carcinoma": "cancer",
#         "squamous cell": "cell",
#         "carcinoma": "cancer",
#     }
#     for k, v in replacement.items(): 
#         if k in ctx:   
#             ctx = ctx.replace(k, v)  
#     return " ".join([w for w in ctx.split(" ") if w])

def parse_str(raw):
    """shared by concept names and prompt descriptions"""
    return raw.strip("-").strip()

def parse_kv(label, raw):
    key = raw.split(":")[0].strip()
    value = "".join(raw.split(":")[1:])
    # value = remove_class_name(class_name, value).strip()
    return key, value

def class_abbr(name):
    return "-".join(name.lower().split(" "))[:10]

def parse_prompt(src_file, output_dir):
    patch_ctx = dict()
    slide_ctx = dict()
    with open(src_file, "r") as src:
        for line in src.readlines():
            cols = line.strip().lower().split(",")
            prompt_level = cols[0]
            label = cols[1]
            prompt_desc = "".join(cols[2:])
            if prompt_level == "patch":
                if label not in patch_ctx:
                    patch_ctx[label] = {}
                key, value = parse_kv(label, prompt_desc)
                key = parse_str(key)
                patch_ctx[label][key] = parse_str(value)
            else:
                slide_ctx[label] = parse_str(prompt_desc)
            
    # filter TOP-N concepts
    for class_name, ctx in patch_ctx.items():
        kv_pairs = [(k,v) for k,v in ctx.items()][:NUM_PROTO]
        patch_ctx[class_name] = {k:v for (k,v) in kv_pairs}
                  
    with open(os.path.join(output_dir, "slide_prompts.json"), "w") as dst:
        dst.write(json.dumps(slide_ctx, indent=2))
        
    with open(os.path.join(output_dir, "patch_prompts.json"), "w") as dst:
        dst.write(json.dumps(patch_ctx, indent=2))

def main():
    print("parsing prompts..")
    parse_prompt(
        src_file=f"./experiment/{EXPERIMENT_NAME}/input/prompt/gpt4-prompts_raw.txt", 
        output_dir=f"./experiment/{EXPERIMENT_NAME}/input/prompt/"
    )
    print("done!")

if __name__ == "__main__":
    main()
