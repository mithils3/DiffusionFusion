from datasets import Dataset, Features, Array3D, Value

features = Features({
    "latent": Array3D(shape=(4, 32, 32), dtype="float32"),  # or float32
    "label": Value("int64"),
    "id": Value("int64"),
})

