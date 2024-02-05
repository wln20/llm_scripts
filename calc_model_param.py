"""
Description: Calculate the number of a model's parameters
"""

# model instantialization
# model = XXX

total = sum(p.numel() for p in model.parameters())
print("Total model params: %.2fM" % (total / 1e6))
