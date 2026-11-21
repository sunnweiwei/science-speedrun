# Native Low measurements

On RTX A6000, the public paper-size run generated 150 unconditional monomers, 50 per length. AF3 passed 119/150. Foldseek yielded **82/150 distinct passing clusters (54.7%)**. End-to-end yield including final AF3 was 112.5 distinct passing clusters per GPU-hour. Each length used generation seed 727000 + length and AF3 seed 727000.

[100 residues](reproduction/length-100.json) · [150 residues](reproduction/length-150.json) · [250 residues](reproduction/length-250.json) · [Generation log](reproduction/generation.log)
